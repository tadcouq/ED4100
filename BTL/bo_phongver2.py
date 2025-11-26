import streamlit as st
import simpy
import random
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from datetime import datetime, timedelta

# ==========================================
# 0. CẤU HÌNH & HÀM TIỆN ÍCH
# ==========================================
st.set_page_config(page_title="Digital Twin Park V2", layout="wide")

# Hàm chuyển đổi giờ (HH:MM) sang phút mô phỏng (int)
def time_to_min(time_obj, start_time_obj):
    delta = datetime.combine(datetime.today(), time_obj) - datetime.combine(datetime.today(), start_time_obj)
    return int(delta.total_seconds() / 60)

# Hàm chuyển ngược phút sang giờ string
def min_to_time_str(minutes, start_time_obj):
    new_time = datetime.combine(datetime.today(), start_time_obj) + timedelta(minutes=minutes)
    return new_time.strftime("%H:%M")

# ==========================================
# 1. INPUT MODULE (SIDEBAR & CONFIG)
# ==========================================
st.title("🎡 Digital Twin V2: Advanced Flow & Network Simulation")
st.markdown("---")

with st.sidebar:
    st.header("⚙️ Cấu hình Vận hành")
    
    # 1.1 Thời gian & Sức chứa
    col_t1, col_t2 = st.columns(2)
    OPEN_TIME = col_t1.time_input("Giờ Mở cửa", value=datetime.strptime("08:00", "%H:%M").time())
    CLOSE_TIME = col_t2.time_input("Giờ Đóng cửa", value=datetime.strptime("18:00", "%H:%M").time())
    
    # Tính tổng thời gian vận hành
    dummy_date = datetime.today()
    t1 = datetime.combine(dummy_date, OPEN_TIME)
    t2 = datetime.combine(dummy_date, CLOSE_TIME)
    TOTAL_MINUTES = int((t2 - t1).total_seconds() / 60)
    
    STOP_ENTRY_MINUTES = st.number_input("Ngưng nhận khách trước đóng cửa (phút)", value=90)
    AVG_DWELL_TIME = st.number_input("Thời gian lưu trú TB (phút)", value=180)
    PARK_CAPACITY = st.number_input("Sức chứa Công viên", value=3000)
    TOTAL_VISITORS = st.number_input("Tổng khách dự kiến", value=2000)

    st.markdown("---")
    st.header("🎫 Vé & Hành vi")
    
    # 1.2 Vé Combo vs Lẻ
    col_v1, col_v2 = st.columns(2)
    RATIO_COMBO = col_v1.slider("Tỷ lệ Vé Combo (%)", 0, 100, 40, help="Đã bao gồm dịch vụ")
    RATIO_SINGLE = 100 - RATIO_COMBO
    col_v2.info(f"Vé Lẻ: {RATIO_SINGLE}%")
    
    TICKET_PRICE_COMBO = st.number_input("Giá Vé Combo (VNĐ)", value=500000)
    TICKET_PRICE_ENTRY = st.number_input("Giá Vé Cổng (cho khách lẻ)", value=100000)

# --- MAIN AREA: CẤU HÌNH KHU DỊCH VỤ ĐỘNG ---
st.subheader("🛠️ Cấu hình Các Khu Dịch vụ (Service Nodes)")
st.info("💡 Bạn có thể thêm/sửa/xóa các khu vui chơi ngay tại bảng dưới đây.")

# Dữ liệu mặc định
default_nodes = [
    {"Tên Khu": "Tàu lượn", "Loại": "Trò chơi", "Nhân viên": 3, "Tốc độ (phút)": 5, "Sức chứa hàng đợi": 50, "Giá/Chi tiêu (VNĐ)": 50000, "Tỷ lệ quay lại (%)": 10},
    {"Tên Khu": "Nhà hàng", "Loại": "Ăn uống", "Nhân viên": 5, "Tốc độ (phút)": 30, "Sức chứa hàng đợi": 100, "Giá/Chi tiêu (VNĐ)": 150000, "Tỷ lệ quay lại (%)": 5},
    {"Tên Khu": "Đu quay", "Loại": "Trò chơi", "Nhân viên": 2, "Tốc độ (phút)": 8, "Sức chứa hàng đợi": 30, "Giá/Chi tiêu (VNĐ)": 30000, "Tỷ lệ quay lại (%)": 15},
]

edited_nodes_df = st.data_editor(pd.DataFrame(default_nodes), num_rows="dynamic", use_container_width=True)

# ==========================================
# 2. SIMULATION ENGINE (CORE LOGIC)
# ==========================================

class ServiceNode:
    """Đại diện cho một khu vui chơi/dịch vụ"""
    def __init__(self, env, name, config):
        self.env = env
        self.name = name
        self.resource = simpy.Resource(env, capacity=int(config["Nhân viên"]))
        self.service_time = config["Tốc độ (phút)"]
        self.queue_cap = config["Sức chứa hàng đợi"]
        self.price = config["Giá/Chi tiêu (VNĐ)"]
        self.rebuy_prob = config["Tỷ lệ quay lại (%)"] / 100.0
        self.revenue = 0
        self.visits = 0

class DigitalTwinPark:
    def __init__(self, env, nodes_config):
        self.env = env
        self.nodes = {}
        # Khởi tạo dynamic nodes từ config
        for idx, row in nodes_config.iterrows():
            self.nodes[row["Tên Khu"]] = ServiceNode(env, row["Tên Khu"], row)
            
        self.gate_revenue = 0
        self.current_visitors = 0
        
        # Tracking logs
        self.visitor_paths = [] # Lưu luồng đi: [Visitor_ID, Node_From, Node_To]
        self.snapshots = [] # Lưu vị trí khách tại mỗi khung giờ (cho Animation)

    def capture_snapshot(self):
        """Chụp ảnh vị trí khách mỗi 30 phút để làm Animation"""
        while True:
            # Snapshot đơn giản hóa: Random vị trí dựa trên current visitors
            # Trong thực tế, cần tracking từng object agent
            snapshot_time = min_to_time_str(self.env.now, OPEN_TIME)
            
            # Ghi nhận trạng thái hàng đợi của từng node
            for name, node in self.nodes.items():
                queue_len = len(node.resource.queue)
                processing = node.resource.count
                self.snapshots.append({
                    "Time": snapshot_time,
                    "Node": name,
                    "Visitors": queue_len + processing,
                    "Type": "Service"
                })
            
            # Ghi nhận tại cổng/đường đi (giả lập số dư)
            visitors_in_nodes = sum([len(n.resource.queue) + n.resource.count for n in self.nodes.values()])
            walking = max(0, self.current_visitors - visitors_in_nodes)
            self.snapshots.append({
                "Time": snapshot_time,
                "Node": "Walking/Path",
                "Visitors": walking,
                "Type": "Path"
            })
            
            yield self.env.timeout(30) # 30 phút chụp 1 lần

def visitor_journey(env, visitor_id, park, is_combo, entry_time):
    """Hành trình của khách hàng: Cổng -> Chọn Node -> Chơi -> (Lặp lại) -> Ra về"""
    
    # 1. Check-in Cổng
    park.current_visitors += 1
    if is_combo:
        park.gate_revenue += TICKET_PRICE_COMBO
    else:
        park.gate_revenue += TICKET_PRICE_ENTRY
    
    current_location = "Cổng vào"
    
    # Vòng lặp trải nghiệm (Dựa trên thời gian lưu trú)
    # Khách sẽ đi khoảng 2-4 điểm dịch vụ trong thời gian lưu trú
    stay_duration = random.gauss(AVG_DWELL_TIME, 30)
    leave_time = entry_time + stay_duration
    
    # Danh sách các khu để chọn ngẫu nhiên
    node_names = list(park.nodes.keys())
    
    while env.now < leave_time and env.now < TOTAL_MINUTES:
        # Chọn điểm đến tiếp theo (Random đơn giản, có thể nâng cấp thành logic độ hấp dẫn)
        if not node_names: break
        target_name = random.choice(node_names)
        target_node = park.nodes[target_name]
        
        # Di chuyển (Walking time)
        walk_time = random.randint(5, 15)
        yield env.timeout(walk_time)
        
        # LOGGING SANKEY: Di chuyển từ Node cũ -> Node mới
        park.visitor_paths.append({
            "Source": current_location, 
            "Target": target_name, 
            "Value": 1
        })
        current_location = target_name
        
        # -- QUY TRÌNH DỊCH VỤ --
        # Kiểm tra hàng đợi (Balking)
        if len(target_node.resource.queue) < target_node.queue_cap:
            with target_node.resource.request() as req:
                yield req # Xếp hàng
                yield env.timeout(target_node.service_time) # Sử dụng dịch vụ
                
                # Tính doanh thu dịch vụ
                # Logic: Khách Combo không mất tiền vé lẻ (trừ khi là Nhà hàng/Shop nếu logic quy định khác)
                # Ở đây giả định: Combo free vé trò chơi, nhưng vẫn mất tiền ăn. 
                # Để đơn giản: Combo free tất cả trừ khi Re-buy
                
                # Logic thanh toán:
                pay_amount = 0
                if not is_combo:
                    pay_amount = target_node.price
                
                target_node.revenue += pay_amount
                target_node.visits += 1
                
                # -- LOGIC RE-LOOP (QUAY LẠI MUA THÊM) --
                # Khách chơi xong, có muốn chơi lại ngay không?
                if random.random() < target_node.rebuy_prob:
                    # Quay lại xếp hàng
                    yield env.timeout(2) # Đi vòng lại
                    # Lần này chắc chắn phải trả tiền (Doanh thu lặp lại)
                    target_node.revenue += target_node.price 
                    # Log path quay lại chính nó
                    park.visitor_paths.append({"Source": target_name, "Target": target_name, "Value": 1})

        else:
            # Bỏ qua do hàng dài
            pass
            
    # Ra về
    park.current_visitors -= 1
    park.visitor_paths.append({
        "Source": current_location, 
        "Target": "Ra về", 
        "Value": 1
    })

def park_generator(env, park, total_visitors):
    """Sinh khách dựa trên thời gian thực"""
    # Ngưng nhận khách trước giờ đóng cửa
    stop_entry_time = TOTAL_MINUTES - STOP_ENTRY_MINUTES
    
    # Tốc độ sinh khách (Inter-arrival time)
    # Giả lập phân phối hình chuông (đông trưa)
    
    visitor_count = 0
    while env.now < stop_entry_time and visitor_count < total_visitors:
        # Kiểm tra sức chứa
        if park.current_visitors < PARK_CAPACITY:
            visitor_count += 1
            is_combo = random.random() < (RATIO_COMBO / 100.0)
            env.process(visitor_journey(env, f"Vis_{visitor_count}", park, is_combo, env.now))
        
        # Random thời gian khách đến tiếp theo
        # Đơn giản hóa: đến ngẫu nhiên
        yield env.timeout(random.expovariate(1.0 / (TOTAL_MINUTES/total_visitors)))

# ==========================================
# 3. VISUALIZATION FUNCTIONS
# ==========================================

def draw_sankey(path_data):
    """Vẽ Sankey Diagram từ log di chuyển"""
    if not path_data:
        return None
        
    df = pd.DataFrame(path_data)
    # Tổng hợp số lượng di chuyển giữa các cặp Source-Target
    df_aggr = df.groupby(["Source", "Target"]).size().reset_index(name="Count")
    
    # Tạo danh sách node duy nhất
    all_nodes = list(pd.concat([df_aggr["Source"], df_aggr["Target"]]).unique())
    node_map = {name: i for i, name in enumerate(all_nodes)}
    
    link_source = df_aggr["Source"].map(node_map).tolist()
    link_target = df_aggr["Target"].map(node_map).tolist()
    link_value = df_aggr["Count"].tolist()
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15, thickness=20, line=dict(color="black", width=0.5),
            label=all_nodes,
            color="blue"
        ),
        link=dict(
            source=link_source, target=link_target, value=link_value
        ))])
    fig.update_layout(title_text="Luồng di chuyển của Khách hàng (Sankey Flow)", font_size=10)
    return fig

def draw_network_animation(nodes_list, snapshots):
    """Vẽ Network Animation (Scatter Plot trên nền Graph)"""
    if not snapshots: return None
    
    # 1. Tạo Graph Layout cố định
    G = nx.Graph()
    G.add_node("Cổng vào")
    G.add_node("Ra về")
    G.add_node("Walking/Path")
    for n in nodes_list:
        G.add_node(n["Tên Khu"])
    
    # Tạo liên kết giả để vẽ layout đẹp (Star topology)
    for n in nodes_list:
        G.add_edge("Cổng vào", "Walking/Path")
        G.add_edge("Walking/Path", n["Tên Khu"])
        G.add_edge(n["Tên Khu"], "Ra về")
    
    pos = nx.spring_layout(G, seed=42)
    
    # 2. Chuẩn bị DataFrame cho Plotly Animation
    df_anim = pd.DataFrame(snapshots)
    
    # Map tọa độ vào DataFrame
    df_anim["x"] = df_anim["Node"].map(lambda n: pos[n][0] if n in pos else 0)
    df_anim["y"] = df_anim["Node"].map(lambda n: pos[n][1] if n in pos else 0)
    
    # Vẽ Bubble Chart Animation
    fig = px.scatter(
        df_anim, x="x", y="y", 
        animation_frame="Time", animation_group="Node",
        size="Visitors", color="Node", 
        hover_name="Node", size_max=60,
        range_x=[-1.5, 1.5], range_y=[-1.5, 1.5],
        title="Mô phỏng Mật độ Khách theo Thời gian thực"
    )
    
    # Vẽ thêm các cạnh (Edges) nền tĩnh
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y, line=dict(width=1, color='#888'), hoverinfo='none', mode='lines'
    ))
    
    fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 500 # Tốc độ animation
    return fig

# ==========================================
# 4. RUN & DASHBOARD DISPLAY
# ==========================================

if st.button("🚀 CHẠY MÔ PHỎNG CHI TIẾT", type="primary"):
    # 1. Setup Environment
    env = simpy.Environment()
    park = DigitalTwinPark(env, edited_nodes_df)
    
    # 2. Register Processes
    env.process(park_generator(env, park, TOTAL_VISITORS))
    env.process(park.capture_snapshot()) # Tracking cho animation
    
    # 3. Run
    with st.spinner("Đang xử lý hàng nghìn tác vụ mô phỏng..."):
        env.run(until=TOTAL_MINUTES)
    
    # 4. Display Results
    st.success("Mô phỏng hoàn tất!")
    
    # --- TAB 1: FLOW VISUALIZATION (Yêu cầu mới) ---
    tab_flow, tab_fin, tab_data = st.tabs(["🌊 Luồng Khách (Flow)", "💰 Doanh thu", "📋 Dữ liệu thô"])
    
    with tab_flow:
        st.write("### 1. Sankey Diagram: Hành trình Khách hàng")
        st.caption("Biểu đồ thể hiện dòng chảy từ lúc vào cổng -> qua các khu dịch vụ -> ra về/quay lại.")
        fig_sankey = draw_sankey(park.visitor_paths)
        if fig_sankey:
            st.plotly_chart(fig_sankey, use_container_width=True)
        
        st.write("### 2. Network Animation: Mật độ theo Giờ")
        st.caption("Bấm nút 'Play' bên dưới để xem sự di chuyển/tích tụ của khách theo thời gian.")
        fig_anim = draw_network_animation(default_nodes, park.snapshots) # Lưu ý: default_nodes ở đây cần update từ edited_df nếu user sửa tên
        if fig_anim:
            st.plotly_chart(fig_anim, use_container_width=True)

    # --- TAB 2: FINANCIALS ---
    with tab_fin:
        # Tính tổng doanh thu dịch vụ phụ
        service_revenue = sum([n.revenue for n in park.nodes.values()])
        total_rev = park.gate_revenue + service_revenue
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Tổng Doanh Thu", f"{total_rev:,.0f} VNĐ")
        c2.metric("Doanh thu Vé Cổng", f"{park.gate_revenue:,.0f} VNĐ")
        c3.metric("Doanh thu Dịch vụ Phụ", f"{service_revenue:,.0f} VNĐ")
        
        st.write("#### Chi tiết Doanh thu từng khu (Bao gồm Re-buy)")
        rev_data = []
        for name, node in park.nodes.items():
            rev_data.append({"Khu vực": name, "Doanh thu": node.revenue, "Lượt khách": node.visits})
        st.bar_chart(pd.DataFrame(rev_data).set_index("Khu vực")["Doanh thu"])

    with tab_data:
        st.dataframe(pd.DataFrame(park.visitor_paths).head(100))

else:
    st.info("Hãy cấu hình các khu dịch vụ ở bảng trên và bấm nút Chạy.")