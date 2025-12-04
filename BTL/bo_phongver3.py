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
st.set_page_config(page_title="Digital Twin Park V5 (Final)", layout="wide")

DEFAULT_OPEN_TIME = datetime.strptime("08:00", "%H:%M").time()
COORD_LOG_INTERVAL = 5 # Log vị trí mỗi 5 phút mô phỏng

def time_to_min(time_obj, start_time_obj):
    delta = datetime.combine(datetime.today(), time_obj) - datetime.combine(datetime.today(), start_time_obj)
    return int(delta.total_seconds() / 60)

def min_to_time_str(minutes, start_time_obj):
    new_time = datetime.combine(datetime.today(), start_time_obj) + timedelta(minutes=minutes)
    return new_time.strftime("%H:%M")

def create_park_layout(nodes_list):
    """Tạo layout (tọa độ cố định) cho các nodes để vẽ Network và Dots"""
    G = nx.Graph()
    G.add_nodes_from(["Cổng vào", "Ra về", "Đường đi/Khác"])
    for n in nodes_list: G.add_node(n["Tên Khu"])
    
    # Tạo liên kết giả để vẽ layout
    for n in nodes_list:
        G.add_edge("Cổng vào", n["Tên Khu"], weight=1)
        G.add_edge(n["Tên Khu"], "Ra về", weight=1)
    
    # Trả về tọa độ cố định
    return nx.spring_layout(G, k=0.8, iterations=50, seed=42)

# ==========================================
# 1. CLASS DEFINITIONS (Tài nguyên và Trạng thái)
# ==========================================

class ServiceNode:
    def __init__(self, env, name, config, pos):
        self.env = env
        self.name = name
        cap = int(config["Nhân viên"])
        if config["Loại"] == "Cảnh quan": cap = 9999 # Unlimited for passive nodes
            
        self.resource = simpy.Resource(env, capacity=cap)
        self.service_time = config["Tốc độ (phút)"]
        self.queue_cap = config["Sức chứa hàng đợi"]
        self.price = config["Giá/Chi tiêu (VNĐ)"]
        self.rebuy_prob = config["Tỷ lệ quay lại (%)"] / 100.0
        self.failure_rate = config["Tỷ lệ hỏng (%)"]
        self.mttr = config["TG Sửa (phút)"]
        self.revenue = 0; self.visits = 0; self.breakdown_count = 0
        self.pos = pos[name] 
        
        if self.failure_rate > 0: self.env.process(self.breakdown_control())

    def breakdown_control(self):
        while True:
            if self.failure_rate > 0:
                time_to_fail = random.expovariate(self.failure_rate / 1000.0) 
                yield self.env.timeout(time_to_fail)
                self.breakdown_count += 1
                reqs = [self.resource.request() for _ in range(self.resource.capacity)]
                yield simpy.AllOf(self.env, reqs)
                yield self.env.timeout(self.mttr)
                for req in reqs: self.resource.release(req)
            else:
                yield self.env.timeout(999999)

class DigitalTwinPark:
    def __init__(self, env, nodes_config, pos):
        self.env = env
        self.nodes_pos = pos
        self.nodes = {}
        for idx, row in nodes_config.iterrows():
            self.nodes[row["Tên Khu"]] = ServiceNode(env, row["Tên Khu"], row, pos)
            
        self.gate_qr = simpy.Resource(env, capacity=4) 
        self.gate_booking = simpy.Resource(env, capacity=2)
        self.gate_walkin = simpy.Resource(env, capacity=2)

        self.gate_revenue = 0
        self.current_visitors = 0
        self.snapshots = []       # Dành cho Heatmap (Aggregate)
        self.agent_tracker = []   # Dành cho Moving Dots (Individual)
        self.last_log_time = -1
        self.unique_agents = set()

    def log_position(self, visitor_id, current_node, status, is_combo, x=None, y=None):
        """Ghi log vị trí cá nhân cho hoạt ảnh Moving Dots"""
        if self.env.now > self.last_log_time or self.env.now == 0:
             self.last_log_time = self.env.now
             
        self.agent_tracker.append({
            'Time': min_to_time_str(self.env.now, DEFAULT_OPEN_TIME),
            'ID': visitor_id,
            'x': x if x is not None else self.nodes_pos[current_node][0],
            'y': y if y is not None else self.nodes_pos[current_node][1],
            'Status': status,
            'Is_Combo': is_combo
        })
        self.unique_agents.add(visitor_id) # Track số lượng agents

    def capture_snapshot(self):
        """Ghi lại trạng thái tổng hợp cho Heatmap"""
        while True:
            snapshot_time = min_to_time_str(self.env.now, DEFAULT_OPEN_TIME)
            visitors_in_nodes = 0
            for name, node in self.nodes.items():
                count = len(node.resource.queue) + node.resource.count
                visitors_in_nodes += count
                self.snapshots.append({
                    "Time": snapshot_time, "Node": name, "Visitors": count
                })
            walking = max(0, self.current_visitors - visitors_in_nodes)
            self.snapshots.append({
                "Time": snapshot_time, "Node": "Đường đi/Khác", "Visitors": walking
            })
            yield self.env.timeout(30) # Log mỗi 30 phút

# ==========================================
# 2. PROCESS DEFINITIONS (FIX LỖI NAMERROR)
# ==========================================

def visitor_journey(env, visitor_id, park, is_combo, entry_time):
    # --- 1. CHECK-IN (Giữ nguyên logic V3) ---
    rand_gate = random.random() * 100
    if rand_gate < GATE_QR_PCT:
        with park.gate_qr.request() as req: yield req; yield env.timeout(0.5)
    elif rand_gate < GATE_QR_PCT + GATE_BOOKING_PCT:
        with park.gate_booking.request() as req: yield req; yield env.timeout(2.0)
    else:
        with park.gate_walkin.request() as req: yield req; yield env.timeout(5.0)

    park.current_visitors += 1
    park.log_position(visitor_id, "Cổng vào", "Entering", is_combo) # Log vị trí khởi tạo
    
    if is_combo: park.gate_revenue += TICKET_PRICE_COMBO
    else: park.gate_revenue += TICKET_PRICE_ENTRY
    
    # --- 2. VÒNG LẶP TRẢI NGHIỆM (Spatial Flow Logic) ---
    current_location = "Cổng vào"
    stay_duration = random.gauss(AVG_DWELL_TIME, 30)
    leave_time = entry_time + stay_duration
    node_names = list(park.nodes.keys())
    
    while env.now < leave_time and env.now < TOTAL_MINUTES:
        target_name = random.choice(node_names)
        target_node = park.nodes[target_name]
        
        # PHASE 1: WALKING (LOG TỌA ĐỘ NỘI SUY)
        walk_time = random.randint(5, 15)
        start_pos = park.nodes_pos[current_location]
        end_pos = park.nodes_pos[target_name]
        
        # Nội suy vị trí (Log từng bước)
        for t_step in np.linspace(0, walk_time, num=int(walk_time / COORD_LOG_INTERVAL) + 1):
            if env.now + t_step >= leave_time: break 
            ratio = t_step / walk_time
            current_x = start_pos[0] * (1 - ratio) + end_pos[0] * ratio
            current_y = start_pos[1] * (1 - ratio) + end_pos[1] * ratio
            park.log_position(visitor_id, "Đường đi/Khác", "Walking", is_combo, x=current_x, y=current_y)
            yield env.timeout(COORD_LOG_INTERVAL)

        current_location = target_name # Đã đến đích

        # PHASE 2: SERVICE
        if len(target_node.resource.queue) < target_node.queue_cap:
            with target_node.resource.request() as req:
                yield req 
                park.log_position(visitor_id, target_name, "Serving", is_combo) # Log đang chơi
                yield env.timeout(target_node.service_time)
                
                # Thanh toán/Rebuy
                if target_node.price > 0 and not is_combo: park.nodes[target_name].revenue += target_node.price
                elif target_node.price > 0 and is_combo and random.random() < target_node.rebuy_prob: park.nodes[target_name].revenue += target_node.price
                park.nodes[target_name].visits += 1
        else:
            park.log_position(visitor_id, current_location, "Balking", is_combo) 
            yield env.timeout(2) # Chờ 2 phút rồi bỏ

    # Ra về
    park.log_position(visitor_id, "Ra về", "Exit", is_combo)
    park.current_visitors -= 1

def park_generator(env, park, total_visitors):
    """Sinh khách lẻ dựa trên thời gian thực"""
    stop_entry_time = TOTAL_MINUTES - STOP_ENTRY_MINUTES
    visitor_count = 0
    while env.now < stop_entry_time and visitor_count < total_visitors:
        if park.current_visitors < PARK_CAPACITY:
            visitor_count += 1
            is_combo = random.random() < (RATIO_COMBO / 100.0)
            env.process(visitor_journey(env, f'Vis_{visitor_count}', park, is_combo, env.now))
        yield env.timeout(random.expovariate(1.0 / (TOTAL_MINUTES/total_visitors)))

def tour_generator(env, park, tours_df):
    """Sinh khách đoàn theo lịch trình"""
    tours = tours_df.to_dict('records')
    for tour in tours: tour['arrival_min'] = time_to_min(tour['Giờ đến'], DEFAULT_OPEN_TIME)
    tours.sort(key=lambda x: x['arrival_min'])
    for tour in tours:
        if tour['arrival_min'] > env.now:
            yield env.timeout(tour['arrival_min'] - env.now)
        for i in range(tour['Số lượng']):
            env.process(visitor_journey(env, f"Tour_{tour['Loại đoàn']}_{i}", park, is_combo=True, entry_time=env.now))

# ==========================================
# 3. VISUALIZATION FUNCTIONS
# ==========================================

def draw_moving_dots(agent_tracker_df, pos):
    """Vẽ hoạt ảnh di chuyển chấm con"""
    if agent_tracker_df.empty: return st.warning("Không có dữ liệu di chuyển.")
    
    # Vẽ nền (Các node cố định)
    fig = go.Figure()
    edge_x = []; edge_y = []
    G = nx.Graph()
    for name in pos.keys(): G.add_node(name)
    # Thêm các cạnh (edges) cho nền tĩnh
    for name in pos.keys():
        if name != "Cổng vào" and name != "Ra về":
            G.add_edge("Cổng vào", name)
            G.add_edge(name, "Ra về")
            
    for edge in G.edges():
        u, v = edge
        if u in pos and v in pos:
            x0, y0 = pos[u]; x1, y1 = pos[v]
            edge_x.extend([x0, x1, None]); edge_y.extend([y0, y1, None])
            
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines', name='Path'))

    # Vẽ Hoạt ảnh các chấm con
    fig = px.scatter(
        agent_tracker_df, x='x', y='y', 
        animation_frame='Time',
        animation_group='ID',
        color='Is_Combo', 
        symbol='Status',
        hover_data=['ID', 'Status'], 
        size_max=15, 
        range_x=[-1.5, 1.5], range_y=[-1.5, 1.5],
        title="Trực quan hóa Vị trí & Dòng Khách Cá nhân (Moving Dots)"
    )
    
    fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 500
    fig.update_traces(marker=dict(size=10, opacity=0.8))
    fig.update_layout(xaxis_visible=False, yaxis_visible=False, showlegend=True)
    
    return fig

# ==========================================
# 4. RUN EXECUTION
# ==========================================

if st.button("🚀 CHẠY MÔ PHỎNG V5 (MOVING DOTS)", type="primary"):
    
    # 1. Lấy Tọa độ cố định
    node_pos_map = create_park_layout(edited_nodes_df.to_dict('records'))
    
    # 2. Setup Environment
    env = simpy.Environment()
    park = DigitalTwinPark(env, edited_nodes_df, node_pos_map)
    
    # 3. Register Processes
    env.process(park_generator(env, park, TOTAL_VISITORS))
    env.process(tour_generator(env, park, edited_tours_df))
    env.process(park.capture_snapshot()) # Vẫn chạy để có dữ liệu Heatmap
    
    with st.spinner("Đang xử lý mô hình không gian..."):
        env.run(until=TOTAL_MINUTES)
    
    st.success("Hoàn tất! Tổng số khách ảo được mô phỏng: " + str(len(park.unique_agents)))
    
    # --- TAB OUTPUT ---
    tab_dots, tab_fin, tab_risk = st.tabs(["🏃 Hoạt ảnh Vị trí", "💰 Tài chính", "⚠️ Rủi ro"])
    
    with tab_dots:
        st.write("### Hoạt ảnh Di chuyển Từng Cá nhân")
        st.caption("Mỗi chấm tròn là một khách hàng. Bấm nút Play (▶️) để xem sự di chuyển và tắc nghẽn.")
        df_dots = pd.DataFrame(park.agent_tracker)
        fig_dots = draw_moving_dots(df_dots, node_pos_map)
        st.plotly_chart(fig_dots, use_container_width=True)

    with tab_fin:
        # ... (Display Financials) ...
        st.metric("Tổng Doanh Thu", f"{park.gate_revenue + sum([n.revenue for n in park.nodes.values()]):,.0f} VNĐ")
        
    with tab_risk:
        # ... (Display Risk Data) ...
        st.write("Dữ liệu rủi ro sẽ hiển thị tại đây.")