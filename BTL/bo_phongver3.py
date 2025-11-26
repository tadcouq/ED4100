import streamlit as st
import simpy
import random
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from datetime import datetime, timedelta, time

# ==========================================
# 0. CẤU HÌNH & HÀM TIỆN ÍCH
# ==========================================
st.set_page_config(page_title="Digital Twin Park V3 (Combined)", layout="wide")

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
st.title("🎡 Digital Twin V3: Hệ thống Mô phỏng Toàn diện")
st.markdown("---")

with st.sidebar:
    st.header("⚙️ 1. Cấu hình Vận hành (Vĩ mô)")
    
    # 1.1 Thời gian & Sức chứa
    col_t1, col_t2 = st.columns(2)
    OPEN_TIME = col_t1.time_input("Giờ Mở cửa", value=datetime.strptime("08:00", "%H:%M").time())
    CLOSE_TIME = col_t2.time_input("Giờ Đóng cửa", value=datetime.strptime("18:00", "%H:%M").time())
    
    # Tính tổng thời gian vận hành
    dummy_date = datetime.today()
    t1 = datetime.combine(dummy_date, OPEN_TIME)
    t2 = datetime.combine(dummy_date, CLOSE_TIME)
    TOTAL_MINUTES = int((t2 - t1).total_seconds() / 60)
    
    STOP_ENTRY_MINUTES = st.number_input("Ngưng nhận khách trước đóng cửa (phút)", value=60)
    AVG_DWELL_TIME = st.number_input("Thời gian lưu trú TB (phút)", value=180)
    PARK_CAPACITY = st.number_input("Sức chứa Công viên", value=3000)
    TOTAL_VISITORS = st.number_input("Tổng khách dự kiến", value=2000)

    st.markdown("---")
    st.header("🎫 2. Vé & Cổng vào")
    
    # 1.2 Vé Combo vs Lẻ
    col_v1, col_v2 = st.columns(2)
    RATIO_COMBO = col_v1.slider("Tỷ lệ Vé Combo (%)", 0, 100, 40, help="Đã bao gồm dịch vụ")
    RATIO_SINGLE = 100 - RATIO_COMBO
    col_v2.info(f"Vé Lẻ: {RATIO_SINGLE}%")
    
    TICKET_PRICE_COMBO = st.number_input("Giá Vé Combo (VNĐ)", value=500000)
    TICKET_PRICE_ENTRY = st.number_input("Giá Vé Cổng (cho khách lẻ)", value=100000)

    # 1.3 Phân luồng Cổng (Từ Ver 1)
    st.subheader("Phân bổ phương thức Check-in")
    col_g1, col_g2, col_g3 = st.columns(3)
    GATE_QR_PCT = col_g1.number_input("% QR Code", value=50)
    GATE_BOOKING_PCT = col_g2.number_input("% Đổi vé Booking", value=30)
    GATE_WALKIN_PCT = col_g3.number_input("% Mua tại quầy", value=20)

# --- MAIN AREA: CẤU HÌNH KHU DỊCH VỤ ĐỘNG (Từ Ver 2 + Breakdown từ Ver 1) ---
st.subheader("🛠️ 3. Cấu hình Các Khu Dịch vụ & Lịch trình")

col_main1, col_main2 = st.columns([2, 1])

with col_main1:
    st.write("**Danh sách Khu vui chơi (Dynamic Nodes)**")
    st.info("💡 Logic Hỏng hóc (MTTR) đã được tích hợp cho từng khu vực.")
    # Dữ liệu mặc định kết hợp tham số breakdown
    default_nodes = [
        {"Tên Khu": "Tàu lượn", "Loại": "Trò chơi", "Nhân viên": 3, "Tốc độ (phút)": 5, "Sức chứa hàng đợi": 50, "Giá/Chi tiêu (VNĐ)": 50000, "Tỷ lệ quay lại (%)": 10, "Tỷ lệ hỏng (%)": 1.0, "TG Sửa (phút)": 30},
        {"Tên Khu": "Nhà hàng", "Loại": "Ăn uống", "Nhân viên": 5, "Tốc độ (phút)": 30, "Sức chứa hàng đợi": 100, "Giá/Chi tiêu (VNĐ)": 150000, "Tỷ lệ quay lại (%)": 5, "Tỷ lệ hỏng (%)": 0.0, "TG Sửa (phút)": 0},
        {"Tên Khu": "Đu quay", "Loại": "Trò chơi", "Nhân viên": 2, "Tốc độ (phút)": 8, "Sức chứa hàng đợi": 30, "Giá/Chi tiêu (VNĐ)": 30000, "Tỷ lệ quay lại (%)": 15, "Tỷ lệ hỏng (%)": 0.5, "TG Sửa (phút)": 20},
    ]
    edited_nodes_df = st.data_editor(pd.DataFrame(default_nodes), num_rows="dynamic", use_container_width=True)

with col_main2:
    st.write("**Lịch trình Khách đoàn (Tour)**")
    # Dữ liệu mẫu Tour (kết hợp logic V1 nhưng dùng giờ V2)
    default_tours = [
        {"Giờ đến": time(9, 0), "Số lượng": 45, "Loại đoàn": "Học sinh"},
        {"Giờ đến": time(14, 30), "Số lượng": 30, "Loại đoàn": "VIP"},
    ]
    edited_tours_df = st.data_editor(
        pd.DataFrame(default_tours),
        num_rows="dynamic",
        column_config={
            "Giờ đến": st.column_config.TimeColumn("Giờ đến", format="HH:mm")
        },
        use_container_width=True
    )

# ==========================================
# 2. SIMULATION ENGINE (CORE LOGIC)
# ==========================================

class ServiceNode:
    """Đại diện cho một khu vui chơi/dịch vụ (Tích hợp logic breakdown)"""
    def __init__(self, env, name, config):
        self.env = env
        self.name = name
        self.resource = simpy.Resource(env, capacity=int(config["Nhân viên"]))
        self.service_time = config["Tốc độ (phút)"]
        self.queue_cap = config["Sức chứa hàng đợi"]
        self.price = config["Giá/Chi tiêu (VNĐ)"]
        self.rebuy_prob = config["Tỷ lệ quay lại (%)"] / 100.0
        
        # Logic Hỏng hóc (Từ Ver 1)
        self.failure_rate = config["Tỷ lệ hỏng (%)"]
        self.mttr = config["TG Sửa (phút)"]
        
        self.revenue = 0
        self.visits = 0
        self.breakdown_count = 0
        
        # Kích hoạt tiến trình gây hỏng hóc riêng cho node này
        if self.failure_rate > 0:
            self.env.process(self.breakdown_control())

    def breakdown_control(self):
        """Tiến trình chạy song song để gây hỏng hóc ngẫu nhiên"""
        while True:
            # Thời gian hoạt động trước khi hỏng (Exponential)
            # failure_rate là %/giờ hoạt động (ví dụ) -> cần convert phù hợp
            # Giả sử rate = 1.0 -> Mean time = 1000 phút (để demo không bị hỏng quá nhiều)
            if self.failure_rate > 0:
                time_to_fail = random.expovariate(self.failure_rate / 1000.0) 
                yield self.env.timeout(time_to_fail)

                # SỰ CỐ XẢY RA
                self.breakdown_count += 1
                
                # Chiếm dụng toàn bộ tài nguyên (Priority request để chặn khách mới)
                # Dùng cách request 'capacity' lần để block toàn bộ server
                reqs = [self.resource.request() for _ in range(self.resource.capacity)]
                yield simpy.AllOf(self.env, reqs) # Chờ khi lấy được hết quyền kiểm soát
                
                # Sửa chữa
                yield self.env.timeout(self.mttr)
                
                # Sửa xong, giải phóng tài nguyên
                for req in reqs:
                    self.resource.release(req)
            else:
                yield self.env.timeout(999999)

class DigitalTwinPark:
    def __init__(self, env, nodes_config):
        self.env = env
        self.nodes = {}
        # Khởi tạo dynamic nodes từ config
        for idx, row in nodes_config.iterrows():
            self.nodes[row["Tên Khu"]] = ServiceNode(env, row["Tên Khu"], row)
            
        # Tài nguyên Cổng (Từ Ver 1)
        self.gate_qr = simpy.Resource(env, capacity=4) 
        self.gate_booking = simpy.Resource(env, capacity=2)
        self.gate_walkin = simpy.Resource(env, capacity=2)

        self.gate_revenue = 0
        self.current_visitors = 0
        
        # Tracking logs (Từ Ver 2)
        self.visitor_paths = [] 
        self.snapshots = [] 

    def capture_snapshot(self):
        """Chụp ảnh vị trí khách mỗi 30 phút để làm Animation"""
        while True:
            snapshot_time = min_to_time_str(self.env.now, OPEN_TIME)
            
            # Ghi nhận trạng thái hàng đợi của từng node
            visitors_in_nodes = 0
            for name, node in self.nodes.items():
                queue_len = len(node.resource.queue)
                processing = node.resource.count
                current_at_node = queue_len + processing
                visitors_in_nodes += current_at_node
                
                self.snapshots.append({
                    "Time": snapshot_time,
                    "Node": name,
                    "Visitors": current_at_node,
                    "Type": "Service"
                })
            
            # Ghi nhận tại cổng/đường đi
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
    
    # --- 1. QUY TRÌNH CHECK-IN (Kết hợp Ver 1) ---
    # Random loại cổng dựa trên tỷ lệ input
    rand_gate = random.random() * 100
    gate_delay = 0
    
    if rand_gate < GATE_QR_PCT:
        with park.gate_qr.request() as req:
            yield req
            gate_delay = 0.5 # 30s
            yield env.timeout(gate_delay)
    elif rand_gate < GATE_QR_PCT + GATE_BOOKING_PCT:
        with park.gate_booking.request() as req:
            yield req
            gate_delay = 2.0 # 2 phút
            yield env.timeout(gate_delay)
    else:
        with park.gate_walkin.request() as req:
            yield req
            gate_delay = 5.0 # 5 phút
            yield env.timeout(gate_delay)

    # Vào được cổng
    park.current_visitors += 1
    if is_combo:
        park.gate_revenue += TICKET_PRICE_COMBO
    else:
        park.gate_revenue += TICKET_PRICE_ENTRY
    
    current_location = "Cổng vào"
    
    # --- 2. VÒNG LẶP TRẢI NGHIỆM (Ver 2) ---
    stay_duration = random.gauss(AVG_DWELL_TIME, 30)
    leave_time = entry_time + gate_delay + stay_duration
    
    node_names = list(park.nodes.keys())
    
    while env.now < leave_time and env.now < TOTAL_MINUTES:
        if not node_names: break
        target_name = random.choice(node_names)
        target_node = park.nodes[target_name]
        
        # Di chuyển
        walk_time = random.randint(5, 15)
        yield env.timeout(walk_time)
        
        # Log path
        park.visitor_paths.append({
            "Source": current_location, "Target": target_name, "Value": 1
        })
        current_location = target_name
        
        # Dùng dịch vụ
        if len(target_node.resource.queue) < target_node.queue_cap:
            with target_node.resource.request() as req:
                yield req # Xếp hàng
                yield env.timeout(target_node.service_time) # Sử dụng dịch vụ
                
                # Thanh toán (Logic Ver 2)
                pay_amount = 0
                if not is_combo:
                    pay_amount = target_node.price
                
                target_node.revenue += pay_amount
                target_node.visits += 1
                
                # Re-loop (Quay lại mua thêm)
                if random.random() < target_node.rebuy_prob:
                    yield env.timeout(2) 
                    target_node.revenue += target_node.price 
                    park.visitor_paths.append({"Source": target_name, "Target": target_name, "Value": 1})
        else:
            # Bỏ qua do hàng dài
            pass
            
    # Ra về
    park.current_visitors -= 1
    park.visitor_paths.append({
        "Source": current_location, "Target": "Ra về", "Value": 1
    })

def park_generator(env, park, total_visitors):
    """Sinh khách lẻ dựa trên thời gian thực"""
    stop_entry_time = TOTAL_MINUTES - STOP_ENTRY_MINUTES
    visitor_count = 0
    
    while env.now < stop_entry_time and visitor_count < total_visitors:
        if park.current_visitors < PARK_CAPACITY:
            visitor_count += 1
            is_combo = random.random() < (RATIO_COMBO / 100.0)
            env.process(visitor_journey(env, f"Vis_{visitor_count}", park, is_combo, env.now))
        
        # Random arrival
        yield env.timeout(random.expovariate(1.0 / (TOTAL_MINUTES/total_visitors)))

def tour_generator(env, park, tours_df):
    """Sinh khách đoàn theo lịch trình (Ver 1 logic adapted to Ver 2 Time)"""
    tours = tours_df.to_dict('records')
    # Convert time object to minutes from start
    for tour in tours:
        tour['arrival_min'] = time_to_min(tour['Giờ đến'], OPEN_TIME)
    
    tours.sort(key=lambda x: x['arrival_min'])
    
    for tour in tours:
        arrival_min = tour['arrival_min']
        count = tour['Số lượng']
        
        if arrival_min > env.now:
            yield env.timeout(arrival_min - env.now)
            
        # Sinh cả nhóm
        for i in range(count):
             # Giả định khách đoàn luôn có vé booking hoặc QR
            env.process(visitor_journey(env, f"Tour_{tour['Loại đoàn']}_{i}", park, is_combo=True, entry_time=env.now))

# ==========================================
# 3. VISUALIZATION FUNCTIONS (Giữ nguyên Ver 2)
# ==========================================

def draw_sankey(path_data):
    if not path_data: return None
    df = pd.DataFrame(path_data)
    df_aggr = df.groupby(["Source", "Target"]).size().reset_index(name="Count")
    all_nodes = list(pd.concat([df_aggr["Source"], df_aggr["Target"]]).unique())
    node_map = {name: i for i, name in enumerate(all_nodes)}
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5), label=all_nodes, color="blue"),
        link=dict(source=df_aggr["Source"].map(node_map), target=df_aggr["Target"].map(node_map), value=df_aggr["Count"])
    )])
    fig.update_layout(title_text="Luồng di chuyển (Sankey Flow)", font_size=10)
    return fig

def draw_network_animation(nodes_list, snapshots):
    if not snapshots: return None
    G = nx.Graph()
    G.add_node("Cổng vào"); G.add_node("Ra về"); G.add_node("Walking/Path")
    for n in nodes_list: G.add_node(n["Tên Khu"])
    for n in nodes_list:
        G.add_edge("Cổng vào", "Walking/Path"); G.add_edge("Walking/Path", n["Tên Khu"]); G.add_edge(n["Tên Khu"], "Ra về")
    pos = nx.spring_layout(G, seed=42)
    
    df_anim = pd.DataFrame(snapshots)
    df_anim["x"] = df_anim["Node"].map(lambda n: pos[n][0] if n in pos else 0)
    df_anim["y"] = df_anim["Node"].map(lambda n: pos[n][1] if n in pos else 0)
    
    fig = px.scatter(df_anim, x="x", y="y", animation_frame="Time", animation_group="Node",
        size="Visitors", color="Node", hover_name="Node", size_max=60, range_x=[-1.5, 1.5], range_y=[-1.5, 1.5],
        title="Mật độ Khách theo Thời gian thực")
    
    edge_x = []; edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]; x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None]); edge_y.extend([y0, y1, None])
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color='#888'), hoverinfo='none', mode='lines'))
    return fig

# ==========================================
# 4. RUN & DASHBOARD DISPLAY
# ==========================================

if st.button("🚀 CHẠY MÔ PHỎNG HỢP NHẤT", type="primary"):
    env = simpy.Environment()
    park = DigitalTwinPark(env, edited_nodes_df)
    
    env.process(park_generator(env, park, TOTAL_VISITORS))
    env.process(tour_generator(env, park, edited_tours_df)) # Tour Generator V3
    env.process(park.capture_snapshot())
    
    with st.spinner("Đang tính toán hàng nghìn khách hàng & sự kiện hỏng hóc..."):
        env.run(until=TOTAL_MINUTES)
    
    st.success("Mô phỏng hoàn tất!")
    
    # --- TAB OUTPUT ---
    tab_flow, tab_fin, tab_risk = st.tabs(["🌊 Luồng & Mạng lưới", "💰 Tài chính & Doanh thu", "⚠️ Rủi ro & Sự cố"])
    
    with tab_flow:
        st.write("### 1. Sankey Diagram")
        fig_sankey = draw_sankey(park.visitor_paths)
        if fig_sankey: st.plotly_chart(fig_sankey, use_container_width=True)
        
        st.write("### 2. Network Animation")
        fig_anim = draw_network_animation(default_nodes, park.snapshots)
        if fig_anim: st.plotly_chart(fig_anim, use_container_width=True)

    with tab_fin:
        service_revenue = sum([n.revenue for n in park.nodes.values()])
        total_rev = park.gate_revenue + service_revenue
        c1, c2, c3 = st.columns(3)
        c1.metric("Tổng Doanh Thu", f"{total_rev:,.0f} VNĐ")
        c2.metric("Doanh thu Vé Cổng", f"{park.gate_revenue:,.0f} VNĐ")
        c3.metric("Doanh thu Dịch vụ", f"{service_revenue:,.0f} VNĐ")
        
        st.write("#### Chi tiết từng khu")
        rev_data = [{"Khu vực": name, "Doanh thu": n.revenue, "Lượt khách": n.visits} for name, n in park.nodes.items()]
        st.dataframe(pd.DataFrame(rev_data))

    with tab_risk:
        st.write("### Báo cáo Sự cố & Bảo trì (Breakdown Log)")
        st.caption("Các khu vực bị dừng hoạt động do sự cố giả lập dựa trên tỷ lệ hỏng.")
        risk_data = [{"Khu vực": name, "Số lần hỏng": n.breakdown_count, "MTTR (phút)": n.mttr} for name, n in park.nodes.items()]
        st.table(pd.DataFrame(risk_data))
        
        # Biểu đồ so sánh lượt hỏng
        fig_risk = px.bar(pd.DataFrame(risk_data), x="Khu vực", y="Số lần hỏng", title="Tần suất sự cố theo khu vực")
        st.plotly_chart(fig_risk, use_container_width=True)