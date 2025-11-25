import streamlit as st
import simpy
import random
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ==========================================
# 1. CẤU HÌNH TRANG & SIDEBAR (INPUT MODULE)
# ==========================================
st.set_page_config(page_title="Digital Twin Khu Vui Chơi", layout="wide")

st.title("🎢 Digital Twin Simulation: Quản lý Vận hành Khu Vui Chơi")
st.markdown("---")

# --- SIDEBAR: CẤU HÌNH VĨ MÔ ---
st.sidebar.header("⚙️ Cấu hình Vĩ mô (Park Level)")

SIM_DURATION = st.sidebar.slider("Thời gian mô phỏng (phút)", 300, 720, 600, help="Ví dụ: 10 tiếng = 600 phút")
PARK_CAPACITY = st.sidebar.number_input("Sức chứa Công viên (Max Capacity)", value=3000, step=100)
TOTAL_VISITORS = st.sidebar.number_input("Tổng khách dự kiến", value=2500, step=50)

st.sidebar.subheader("Luồng Check-in")
col_s1, col_s2, col_s3 = st.sidebar.columns(3)
ratio_qr = col_s1.number_input("% QR/Thẻ từ", value=50, min_value=0, max_value=100)
ratio_booking = col_s2.number_input("% Đổi vé Booking", value=30, min_value=0, max_value=100)
ratio_walkin = col_s3.number_input("% Mua tại quầy", value=20, min_value=0, max_value=100)

# --- SIDEBAR: CẤU HÌNH DỊCH VỤ CON (MICRO LEVEL) ---
st.sidebar.markdown("---")
st.sidebar.header("🛠️ Cấu hình Khu Dịch vụ")

# Ví dụ cấu hình cho 1 khu trò chơi tiêu biểu: Tàu lượn
with st.sidebar.expander("🎢 Khu vực: Tàu lượn siêu tốc", expanded=True):
    RIDE_STAFF = st.slider("Số nhân viên/Server (Fixed Staff)", 1, 10, 3, help="Quyết định công suất phục vụ")
    RIDE_DURATION = st.number_input("Thời gian chơi trung bình (phút)", value=5.0)
    RIDE_PRICE = st.number_input("Giá vé lẻ (VNĐ)", value=50000)
    RIDE_FAILURE_RATE = st.slider("Tỷ lệ hỏng hóc (%)", 0.0, 10.0, 1.0)
    RIDE_MTTR = st.number_input("Thời gian sửa (phút)", value=30)

# --- SIDEBAR: LỊCH TRÌNH KHÁCH ĐOÀN ---
st.sidebar.markdown("---")
st.sidebar.header("🚌 Lịch trình Khách đoàn")

# Dữ liệu mẫu cho bảng
default_tour_data = pd.DataFrame([
    {"Giờ đến (phút)": 60, "Số lượng": 45, "Loại đoàn": "Học sinh"},
    {"Giờ đến (phút)": 120, "Số lượng": 30, "Loại đoàn": "VIP Tour"},
])

tour_schedule = st.sidebar.data_editor(
    default_tour_data,
    num_rows="dynamic",
    use_container_width=True
)

# ==========================================
# 2. LOGIC MÔ PHỎNG (SIMULATION ENGINE)
# ==========================================

class AmusementPark:
    def __init__(self, env, params):
        self.env = env
        self.params = params
        
        # Tài nguyên: Cổng vào (Chia 3 luồng)
        self.gate_qr = simpy.Resource(env, capacity=4) # Giả định 4 cổng tự động
        self.gate_booking = simpy.Resource(env, capacity=2) # 2 quầy đổi vé
        self.gate_walkin = simpy.Resource(env, capacity=2) # 2 quầy thu ngân
        
        # Tài nguyên: Tàu lượn (Số nhân viên cố định từ Input)
        self.ride = simpy.Resource(env, capacity=params['ride_staff'])
        
        # Trạng thái hệ thống
        self.current_visitors = 0
        self.total_revenue = 0
        self.lost_visitors_capacity = 0 # Mất do đầy cổng
        self.lost_visitors_queue = 0    # Mất do chờ lâu
        
        # Log dữ liệu để vẽ biểu đồ
        self.logs = [] 

    def log_status(self):
        """Ghi lại trạng thái hệ thống mỗi 10 phút ảo"""
        while True:
            self.logs.append({
                "time": self.env.now,
                "visitors": self.current_visitors,
                "revenue": self.total_revenue,
                "queue_len": len(self.ride.queue),
                "ride_utilization": self.ride.count / self.ride.capacity
            })
            yield self.env.timeout(10) # Log mỗi 10 phút

    def breakdown_control(self):
        """Quy trình gây hỏng hóc ngẫu nhiên cho Tàu lượn"""
        while True:
            # Thời gian hoạt động trước khi hỏng (Exponential Distribution)
            # Tỷ lệ hỏng 1% -> Mean time to fail = 100 / 1 = 100 giờ (Ví dụ đơn giản hóa)
            if self.params['failure_rate'] > 0:
                time_to_fail = random.expovariate(self.params['failure_rate'] / 1000) 
                yield self.env.timeout(time_to_fail)

                # SỰ CỐ XẢY RA
                repair_time = self.params['mttr']
                
                # Chiếm dụng toàn bộ tài nguyên để sửa (Priority request)
                # Lưu ý: SimPy Resource cơ bản cần PreemptiveResource để ngắt quãng đúng nghĩa
                # Ở đây dùng cách đơn giản: giảm capacity tạm thời
                with self.ride.request() as req:
                    yield req
                    # Giả lập sửa chữa
                    yield self.env.timeout(repair_time)
            else:
                yield self.env.timeout(999999)

def visitor_process(env, name, park, ticket_type, arrival_time):
    """Quy trình hành vi của một khách hàng"""
    
    # 1. KIỂM TRA SỨC CHỨA CÔNG VIÊN (ONE-TIME ENTRY)
    if park.current_visitors >= park.params['max_capacity']:
        park.lost_visitors_capacity += 1
        return # Khách bị chặn, bỏ về ngay

    # Khách được vào -> Tăng biến đếm
    park.current_visitors += 1
    
    # 2. QUY TRÌNH CHECK-IN
    # Chọn luồng dựa trên loại vé
    entry_start = env.now
    if ticket_type == "QR":
        with park.gate_qr.request() as req:
            yield req
            yield env.timeout(0.5) # Nhanh: 30s
            park.total_revenue += 200000 # Giá vé cổng
    elif ticket_type == "Booking":
        with park.gate_booking.request() as req:
            yield req
            yield env.timeout(2.0) # TB: 2 phút
            park.total_revenue += 200000
    else: # Walk-in
        with park.gate_walkin.request() as req:
            yield req
            yield env.timeout(5.0) # Lâu: 5 phút
            park.total_revenue += 250000 # Vé mua tại quầy đắt hơn
    
    # 3. DI CHUYỂN & TRẢI NGHIỆM
    # Đi đến Tàu lượn
    yield env.timeout(random.uniform(5, 15)) # Đi bộ 5-15p
    
    # Quyết định có chơi không? (Dựa trên hàng đợi)
    if len(park.ride.queue) < 50: # Ngưỡng kiên nhẫn: Hàng > 50 người thì bỏ
        with park.ride.request() as req:
            # Chờ xếp hàng
            yield req 
            # Vào chơi
            yield env.timeout(park.params['ride_duration'])
            park.total_revenue += park.params['ride_price'] # Doanh thu dịch vụ phụ
            
            # Re-loop: 20% khách quay lại chơi tiếp (Logic Tái mua)
            if random.random() < 0.2:
                yield env.timeout(2) # Đi vòng lại hàng đợi
                # (Ở đây có thể gọi đệ quy hoặc loop, demo đơn giản là chơi xong rồi nghỉ)
    else:
        park.lost_visitors_queue += 1 # Bỏ do hàng dài

    # 4. LƯU TRÚ & RA VỀ
    # Thời gian chơi các trò khác/ăn uống (Dwell Time)
    dwell_time = random.gauss(180, 30) # TB ở lại 3 tiếng (180p)
    yield env.timeout(dwell_time)
    
    # Khách ra về
    park.current_visitors -= 1

def generator(env, park, total_visitors, duration):
    """Sinh khách lẻ theo thời gian"""
    # Tính tốc độ sinh khách trung bình (khách/phút)
    interarrival = duration / total_visitors 
    
    for i in range(total_visitors):
        yield env.timeout(random.expovariate(1.0 / interarrival))
        
        # Xác định loại vé ngẫu nhiên theo tỷ lệ Input
        rand = random.random() * 100
        if rand < ratio_qr: t_type = "QR"
        elif rand < ratio_qr + ratio_booking: t_type = "Booking"
        else: t_type = "Walkin"
        
        env.process(visitor_process(env, f'Visitor_{i}', park, t_type, env.now))

def tour_generator(env, park, schedule_df):
    """Sinh khách đoàn theo lịch trình"""
    # Convert dataframe to dict list
    tours = schedule_df.to_dict('records')
    # Sort theo giờ đến
    tours.sort(key=lambda x: x['Giờ đến (phút)'])
    
    for tour in tours:
        arrival_time = tour['Giờ đến (phút)']
        count = tour['Số lượng']
        
        # Chờ đến giờ đoàn đến
        if arrival_time > env.now:
            yield env.timeout(arrival_time - env.now)
            
        # Sinh cả nhóm khách cùng lúc (Batch)
        for i in range(count):
            env.process(visitor_process(env, f"Tour_{tour['Loại đoàn']}_{i}", park, "Booking", env.now))

# Hàm chạy chính wrapper
def run_simulation():
    # Gom tham số từ Sidebar
    params = {
        'max_capacity': PARK_CAPACITY,
        'ride_staff': RIDE_STAFF,
        'ride_duration': RIDE_DURATION,
        'ride_price': RIDE_PRICE,
        'failure_rate': RIDE_FAILURE_RATE,
        'mttr': RIDE_MTTR
    }
    
    env = simpy.Environment()
    park = AmusementPark(env, params)
    
    # Kích hoạt các tiến trình
    env.process(park.log_status()) # Ghi log
    env.process(park.breakdown_control()) # Quản lý hỏng hóc
    env.process(generator(env, park, TOTAL_VISITORS, SIM_DURATION)) # Khách lẻ
    env.process(tour_generator(env, park, tour_schedule)) # Khách đoàn
    
    # Chạy mô phỏng
    env.run(until=SIM_DURATION)
    
    return pd.DataFrame(park.logs), park

# ==========================================
# 3. GIAO DIỆN OUTPUT (DASHBOARD MODULE)
# ==========================================

# Nút Action chính
if st.sidebar.button("🚀 CHẠY MÔ PHỎNG (RUN SIMULATION)", type="primary"):
    
    with st.spinner('Hệ thống đang tính toán hàng nghìn khách hàng ảo...'):
        # Gọi hàm chạy
        df_results, park_obj = run_simulation()

    # --- TAB HIỂN THỊ KẾT QUẢ ---
    tab1, tab2, tab3 = st.tabs(["📊 Tổng quan (Overview)", "📈 Phân tích Chi tiết", "💰 Tài chính & Hiệu suất"])

    with tab1:
        st.write("### Kết quả Vận hành trong ngày")
        
        # 4 Chỉ số KPI lớn (Metrics)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Doanh thu Tổng", f"{park_obj.total_revenue:,.0f} VNĐ")
        col2.metric("Khách vào được", f"{df_results['visitors'].max():,.0f} người")
        col3.metric("Mất do Đầy cổng", f"{park_obj.lost_visitors_capacity} người", delta_color="inverse")
        col4.metric("Mất do Hàng dài", f"{park_obj.lost_visitors_queue} người", delta_color="inverse")

        st.write("### Diễn biến Dòng khách (Real-time Capacity)")
        # Vẽ biểu đồ Line Chart: Sức chứa vs Thực tế
        fig_cap = go.Figure()
        fig_cap.add_trace(go.Scatter(x=df_results['time'], y=df_results['visitors'], fill='tozeroy', name='Khách thực tế'))
        fig_cap.add_hline(y=PARK_CAPACITY, line_dash="dash", line_color="red", annotation_text="Giới hạn Sức chứa")
        st.plotly_chart(fig_cap, use_container_width=True)

    with tab2:
        st.write("### Phân tích Điểm nghẽn (Bottlenecks)")
        
        col_c1, col_c2 = st.columns(2)
        
        with col_c1:
            st.write("**Độ dài Hàng đợi Tàu lượn theo thời gian**")
            fig_queue = px.line(df_results, x='time', y='queue_len', labels={'queue_len': 'Số người chờ'})
            st.plotly_chart(fig_queue, use_container_width=True)
            
        with col_c2:
            st.write("**Công suất phục vụ (Utilization)**")
            fig_util = px.area(df_results, x='time', y='ride_utilization', labels={'ride_utilization': '% Bận rộn'})
            st.plotly_chart(fig_util, use_container_width=True)

    with tab3:
        st.write("### Báo cáo Tài chính & Rủi ro")
        
        # Tính toán chi phí cơ hội
        opportunity_cost_gate = park_obj.lost_visitors_capacity * 200000 # Vé TB
        opportunity_cost_queue = park_obj.lost_visitors_queue * RIDE_PRICE
        
        cost_data = pd.DataFrame({
            "Hạng mục": ["Doanh thu Thực", "Mất tại Cổng (Overload)", "Mất tại Dịch vụ (Queue)"],
            "Giá trị": [park_obj.total_revenue, opportunity_cost_gate, opportunity_cost_queue]
        })
        
        fig_pie = px.pie(cost_data, values='Giá trị', names='Hạng mục', title='Cơ cấu Doanh thu & Thất thoát')
        st.plotly_chart(fig_pie, use_container_width=True)

else:
    st.info("Vui lòng điều chỉnh thông số ở thanh bên trái và nhấn nút 'CHẠY MÔ PHỎNG'")