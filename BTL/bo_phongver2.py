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
st.set_page_config(page_title="Digital Twin Park V4 (Heatmap)", layout="wide")

def time_to_min(time_obj, start_time_obj):
    delta = datetime.combine(datetime.today(), time_obj) - datetime.combine(datetime.today(), start_time_obj)
    return int(delta.total_seconds() / 60)

def min_to_time_str(minutes, start_time_obj):
    new_time = datetime.combine(datetime.today(), start_time_obj) + timedelta(minutes=minutes)
    return new_time.strftime("%H:%M")

# ==========================================
# 1. INPUT MODULE
# ==========================================
st.title("🔥 Digital Twin V4: Timeline Heatmap Simulation")
st.markdown("---")

with st.sidebar:
    st.header("⚙️ 1. Cấu hình Vận hành")
    
    col_t1, col_t2 = st.columns(2)
    OPEN_TIME = col_t1.time_input("Giờ Mở cửa", value=datetime.strptime("08:00", "%H:%M").time())
    CLOSE_TIME = col_t2.time_input("Giờ Đóng cửa", value=datetime.strptime("18:00", "%H:%M").time())
    
    dummy_date = datetime.today()
    TOTAL_MINUTES = int((datetime.combine(dummy_date, CLOSE_TIME) - datetime.combine(dummy_date, OPEN_TIME)).total_seconds() / 60)
    
    STOP_ENTRY_MINUTES = st.number_input("Chặn khách trước đóng cửa (phút)", value=60)
    AVG_DWELL_TIME = st.number_input("Thời gian lưu trú TB (phút)", value=180)
    PARK_CAPACITY = st.number_input("Sức chứa Công viên", value=3000)
    TOTAL_VISITORS = st.number_input("Tổng khách dự kiến", value=2000)

    st.markdown("---")
    st.header("🎫 2. Vé & Cổng")
    
    col_v1, col_v2 = st.columns(2)
    RATIO_COMBO = col_v1.slider("Tỷ lệ Vé Combo (%)", 0, 100, 40)
    RATIO_SINGLE = 100 - RATIO_COMBO
    col_v2.info(f"Vé Lẻ: {RATIO_SINGLE}%")
    
    TICKET_PRICE_COMBO = st.number_input("Giá Vé Combo", value=500000)
    TICKET_PRICE_ENTRY = st.number_input("Giá Vé Cổng", value=100000)

    st.subheader("Phân luồng Check-in")
    col_g1, col_g2, col_g3 = st.columns(3)
    GATE_QR_PCT = col_g1.number_input("% QR Code", value=50)
    GATE_BOOKING_PCT = col_g2.number_input("% Booking", value=30)
    GATE_WALKIN_PCT = col_g3.number_input("% Tại quầy", value=20)

# --- MAIN AREA ---
st.subheader("🛠️ 3. Cấu hình Khu vực (Bao gồm Khu Thăm quan)")

col_main1, col_main2 = st.columns([2, 1])

with col_main1:
    st.info("💡 Đã thêm 'Quảng trường' và 'Vườn hoa' để ghi nhận khách thăm quan.")
    # Dữ liệu mặc định ĐÃ CẬP NHẬT thêm khu Cảnh quan
    default_nodes = [
        {"Tên Khu": "Tàu lượn", "Loại": "Trò chơi", "Nhân viên": 3, "Tốc độ (phút)": 5, "Sức chứa hàng đợi": 50, "Giá/Chi tiêu (VNĐ)": 50000, "Tỷ lệ quay lại (%)": 10, "Tỷ lệ hỏng (%)": 1.0, "TG Sửa (phút)": 30},
        {"Tên Khu": "Nhà hàng", "Loại": "Ăn uống", "Nhân viên": 5, "Tốc độ (phút)": 30, "Sức chứa hàng đợi": 100, "Giá/Chi tiêu (VNĐ)": 150000, "Tỷ lệ quay lại (%)": 5, "Tỷ lệ hỏng (%)": 0.0, "TG Sửa (phút)": 0},
        {"Tên Khu": "Đu quay", "Loại": "Trò chơi", "Nhân viên": 2, "Tốc độ (phút)": 8, "Sức chứa hàng đợi": 30, "Giá/Chi tiêu (VNĐ)": 30000, "Tỷ lệ quay lại (%)": 15, "Tỷ lệ hỏng (%)": 0.5, "TG Sửa (phút)": 20},
        # --- KHU VỰC THĂM QUAN (PASSIVE NODES) ---
        {"Tên Khu": "Quảng trường", "Loại": "Cảnh quan", "Nhân viên": 100, "Tốc độ (phút)": 15, "Sức chứa hàng đợi": 1000, "Giá/Chi tiêu (VNĐ)": 0, "Tỷ lệ quay lại (%)": 0, "Tỷ lệ hỏng (%)": 0.0, "TG Sửa (phút)": 0},
        {"Tên Khu": "Vườn hoa", "Loại": "Cảnh quan", "Nhân viên": 100, "Tốc độ (phút)": 20, "Sức chứa hàng đợi": 1000, "Giá/Chi tiêu (VNĐ)": 0, "Tỷ lệ quay lại (%)": 0, "Tỷ lệ hỏng (%)": 0.0, "TG Sửa (phút)": 0},
    ]
    edited_nodes_df = st.data_editor(pd.DataFrame(default_nodes), num_rows="dynamic", use_container_width=True)

with col_main2:
    st.write("**Lịch trình Khách đoàn**")
    default_tours = [
        {"Giờ đến": time(9, 0), "Số lượng": 45, "Loại đoàn": "Học sinh"},
        {"Giờ đến": time(14, 30), "Số lượng": 30, "Loại đoàn": "VIP"},
    ]
    edited_tours_df = st.data_editor(
        pd.DataFrame(default_tours),
        num_rows="dynamic",
        column_config={"Giờ đến": st.column_config.TimeColumn("Giờ đến", format="HH:mm")},
        use_container_width=True
    )

# ==========================================
# 2. SIMULATION ENGINE
# ==========================================

class ServiceNode:
    def __init__(self, env, name, config):
        self.env = env
        self.name = name
        # Nếu là khu cảnh quan, capacity lớn để không bao giờ tắc
        cap = int(config["Nhân viên"])
        if config["Loại"] == "Cảnh quan": cap = 9999
            
        self.resource = simpy.Resource(env, capacity=cap)
        self.service_time = config["Tốc độ (phút)"]
        self.queue_cap = config["Sức chứa hàng đợi"]
        self.price = config["Giá/Chi tiêu (VNĐ)"]
        self.rebuy_prob = config["Tỷ lệ quay lại (%)"] / 100.0
        self.failure_rate = config["Tỷ lệ hỏng (%)"]
        self.mttr = config["TG Sửa (phút)"]
        
        self.revenue = 0
        self.visits = 0
        self.breakdown_count = 0
        
        if self.failure_rate > 0:
            self.env.process(self.breakdown_control())

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
    def __init__(self, env, nodes_config):
        self.env = env
        self.nodes = {}
        for idx, row in nodes_config.iterrows():
            self.nodes[row["Tên Khu"]] = ServiceNode(env, row["Tên Khu"], row)
            
        self.gate_qr = simpy.Resource(env, capacity=4) 
        self.gate_booking = simpy.Resource(env, capacity=2)
        self.gate_walkin = simpy.Resource(env, capacity=2)

        self.gate_revenue = 0
        self.current_visitors = 0
        self.snapshots = [] 

    def capture_snapshot(self):
        """Chụp ảnh Heatmap mỗi 30 phút"""
        while True:
            snapshot_time = min_to_time_str(self.env.now, OPEN_TIME)
            
            visitors_in_nodes = 0
            for name, node in self.nodes.items():
                # Với Heatmap, ta quan tâm tổng số người đang hiện diện ở khu vực đó
                # Bao gồm cả người đang xếp hàng và người đang chơi/ngắm cảnh
                count = len(node.resource.queue) + node.resource.count
                visitors_in_nodes += count
                
                self.snapshots.append({
                    "Time": snapshot_time,
                    "Node": name,
                    "Visitors": count
                })
            
            # Khách đang đi lại giữa các khu (Walking)
            walking = max(0, self.current_visitors - visitors_in_nodes)
            self.snapshots.append({
                "Time": snapshot_time,
                "Node": "Đường đi/Khác",
                "Visitors": walking
            })
            
            yield self.env.timeout(30) 

def visitor_journey(env, visitor_id, park, is_combo, entry_time):
    # --- 1. CHECK-IN ---
    rand_gate = random.random() * 100
    gate_delay = 0
    if rand_gate < GATE_QR_PCT:
        with park.gate_qr.request() as req:
            yield req; yield env.timeout(0.5)
    elif rand_gate < GATE_QR_PCT + GATE_BOOKING_PCT:
        with park.gate_booking.request() as req:
            yield req; yield env.timeout(2.0)
    else:
        with park.gate_walkin.request() as req:
            yield req; yield env.timeout(5.0)

    park.current_visitors += 1
    if is_combo: park.gate_revenue += TICKET_PRICE_COMBO
    else: park.gate_revenue += TICKET_PRICE_ENTRY
    
    # --- 2. DI CHUYỂN ---
    stay_duration = random.gauss(AVG_DWELL_TIME, 30)
    leave_time = entry_time + stay_duration
    node_names = list(park.nodes.keys())
    
    while env.now < leave_time and env.now < TOTAL_MINUTES:
        target_name = random.choice(node_names)
        target_node = park.nodes[target_name]
        
        # Di chuyển
        yield env.timeout(random.randint(5, 15))
        
        # Vào khu vực
        if len(target_node.resource.queue) < target_node.queue_cap:
            with target_node.resource.request() as req:
                yield req
                yield env.timeout(target_node.service_time)
                
                # Trả tiền (nếu không phải vé Combo hoặc khu miễn phí)
                if target_node.price > 0 and not is_combo:
                    target_node.revenue += target_node.price
                elif target_node.price > 0 and is_combo and random.random() < target_node.rebuy_prob:
                     # Rebuy
                     target_node.revenue += target_node.price
                
                target_node.visits += 1

    park.current_visitors -= 1

def park_generator(env, park, total_visitors):
    stop_entry_time = TOTAL_MINUTES - STOP_ENTRY_MINUTES
    visitor_count = 0
    while env.now < stop_entry_time and visitor_count < total_visitors:
        if park.current_visitors < PARK_CAPACITY:
            visitor_count += 1
            is_combo = random.random() < (RATIO_COMBO / 100.0)
            env.process(visitor_journey(env, f"Vis_{visitor_count}", park, is_combo, env.now))
        yield env.timeout(random.expovariate(1.0 / (TOTAL_MINUTES/total_visitors)))

def tour_generator(env, park, tours_df):
    tours = tours_df.to_dict('records')
    for tour in tours: tour['arrival_min'] = time_to_min(tour['Giờ đến'], OPEN_TIME)
    tours.sort(key=lambda x: x['arrival_min'])
    for tour in tours:
        if tour['arrival_min'] > env.now:
            yield env.timeout(tour['arrival_min'] - env.now)
        for i in range(tour['Số lượng']):
            env.process(visitor_journey(env, f"Tour", park, True, env.now))

# ==========================================
# 3. VISUALIZATION: HEATMAP
# ==========================================

def draw_hourly_heatmap(snapshots):
    """Vẽ biểu đồ nhiệt mật độ khách"""
    if not snapshots: return None
    
    df = pd.DataFrame(snapshots)
    
    # Pivot Table: Index=Node, Col=Time, Value=Visitors
    df_pivot = df.pivot_table(index="Node", columns="Time", values="Visitors", aggfunc='sum').fillna(0)
    
    # Sắp xếp lại thứ tự index để "Đường đi" xuống dưới cùng cho đẹp
    # (Optional sorting logic)
    
    # Vẽ Heatmap
    # Dùng thang màu RdYlGn_r (Đỏ - Vàng - Xanh đảo ngược): Đỏ là Đông, Xanh là Vắng
    fig = px.imshow(
        df_pivot,
        labels=dict(x="Thời gian", y="Khu vực", color="Số lượng khách"),
        aspect="auto",
        color_continuous_scale="RdYlGn_r",
        origin='lower'
    )
    fig.update_layout(
        title="🔥 Bản đồ Nhiệt: Mật độ Khách theo Khu vực & Thời gian",
        xaxis_title="Khung giờ trong ngày",
        yaxis_title="Các khu vực (Dịch vụ & Thăm quan)"
    )
    
    # Thêm text hiển thị số lượng lên ô (nếu muốn)
    fig.update_traces(text=df_pivot.values, texttemplate="%{text:.0f}")
    
    return fig

# ==========================================
# 4. RUN
# ==========================================

if st.button("🚀 CHẠY MÔ PHỎNG HEATMAP", type="primary"):
    env = simpy.Environment()
    park = DigitalTwinPark(env, edited_nodes_df)
    
    env.process(park_generator(env, park, TOTAL_VISITORS))
    env.process(tour_generator(env, park, edited_tours_df))
    env.process(park.capture_snapshot())
    
    with st.spinner("Đang tính toán bản đồ nhiệt..."):
        env.run(until=TOTAL_MINUTES)
    
    st.success("Hoàn tất!")
    
    tab_map, tab_fin, tab_risk = st.tabs(["🔥 Bản đồ Nhiệt (Heatmap)", "💰 Doanh thu", "⚠️ Rủi ro"])
    
    with tab_map:
        st.write("### Phân bổ đám đông theo thời gian thực")
        st.caption("Màu đỏ thể hiện khu vực đang quá tải hoặc tập trung đông người (bao gồm cả khách thăm quan).")
        
        fig_heat = draw_hourly_heatmap(park.snapshots)
        if fig_heat: st.plotly_chart(fig_heat, use_container_width=True)
        
        st.write("#### Dữ liệu chi tiết từng khung giờ")
        st.dataframe(pd.DataFrame(park.snapshots).pivot_table(index="Time", columns="Node", values="Visitors", aggfunc='sum').fillna(0))

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
        risk_data = [{"Khu vực": name, "Số lần hỏng": n.breakdown_count, "MTTR (phút)": n.mttr} for name, n in park.nodes.items()]
        fig_risk = px.bar(pd.DataFrame(risk_data), x="Khu vực", y="Số lần hỏng", title="Tần suất sự cố")
        st.plotly_chart(fig_risk, use_container_width=True)