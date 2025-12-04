import streamlit as st
import simpy
import random
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import streamlit.components.v1 as components
from datetime import datetime, timedelta, time

# ==========================================
# 0. CẤU HÌNH & HÀM TIỆN ÍCH
# ==========================================
st.set_page_config(page_title="Digital Twin Park V7 (Analytics)", layout="wide")

def time_to_min(time_obj, start_time_obj):
    """Chuyển đổi thời gian (HH:MM) thành phút tính từ giờ mở cửa"""
    delta = datetime.combine(datetime.today(), time_obj) - datetime.combine(datetime.today(), start_time_obj)
    return int(delta.total_seconds() / 60)

def min_to_time_str(minutes, start_time_obj):
    """Chuyển đổi phút mô phỏng thành chuỗi giờ HH:MM"""
    new_time = datetime.combine(datetime.today(), start_time_obj) + timedelta(minutes=minutes)
    return new_time.strftime("%H:%M")

def min_to_hour_label(minutes, start_time_obj):
    """Lấy nhãn giờ (ví dụ: '09:00', '10:00') cho việc group dữ liệu"""
    new_time = datetime.combine(datetime.today(), start_time_obj) + timedelta(minutes=minutes)
    return new_time.hour

# ==========================================
# 1. INPUT MODULE
# ==========================================
st.title("🔥 Digital Twin V7: Advanced Analytics & Simulation")
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
    TOTAL_VISITORS = st.number_input("Tổng khách dự kiến", value=800) # Tăng nhẹ để số liệu đẹp hơn

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
st.subheader("🛠️ 3. Cấu hình Khu vực & Bản đồ (800x600)")

col_main1, col_main2 = st.columns([2, 1])

with col_main1:
    st.info("💡 Cổng Vào tại (350, 580) - Cổng Ra tại (450, 580).")
    default_nodes = [
        {"Tên Khu": "Tàu lượn", "Loại": "Trò chơi", "Nhân viên": 3, "Tốc độ (phút)": 5, "Sức chứa hàng đợi": 30, "Giá/Chi tiêu (VNĐ)": 50000, "Tỷ lệ hỏng (%)": 0.5, "x": 100, "y": 100},
        {"Tên Khu": "Nhà hàng", "Loại": "Ăn uống", "Nhân viên": 5, "Tốc độ (phút)": 30, "Sức chứa hàng đợi": 50, "Giá/Chi tiêu (VNĐ)": 150000, "Tỷ lệ hỏng (%)": 0.0, "x": 400, "y": 300},
        {"Tên Khu": "Đu quay", "Loại": "Trò chơi", "Nhân viên": 2, "Tốc độ (phút)": 8, "Sức chứa hàng đợi": 20, "Giá/Chi tiêu (VNĐ)": 30000, "Tỷ lệ hỏng (%)": 0.2, "x": 700, "y": 100},
        {"Tên Khu": "Quảng trường", "Loại": "Cảnh quan", "Nhân viên": 100, "Tốc độ (phút)": 15, "Sức chứa hàng đợi": 1000, "Giá/Chi tiêu (VNĐ)": 0, "Tỷ lệ hỏng (%)": 0.0, "x": 400, "y": 500},
        {"Tên Khu": "Vườn hoa", "Loại": "Cảnh quan", "Nhân viên": 100, "Tốc độ (phút)": 20, "Sức chứa hàng đợi": 1000, "Giá/Chi tiêu (VNĐ)": 0, "Tỷ lệ hỏng (%)": 0.0, "x": 700, "y": 500},
        {"Tên Khu": "Khu Quà lưu niệm", "Loại": "Mua sắm", "Nhân viên": 2, "Tốc độ (phút)": 10, "Sức chứa hàng đợi": 15, "Giá/Chi tiêu (VNĐ)": 80000, "Tỷ lệ hỏng (%)": 0.0, "x": 100, "y": 500},
    ]
    edited_nodes_df = st.data_editor(pd.DataFrame(default_nodes), num_rows="dynamic", use_container_width=True)

with col_main2:
    st.write("**Lịch trình Khách đoàn**")
    default_tours = [
        {"Giờ đến": time(9, 0), "Số lượng": 20, "Loại đoàn": "Học sinh"},
        {"Giờ đến": time(14, 30), "Số lượng": 15, "Loại đoàn": "VIP"},
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
    def __init__(self, env, name, config, park_ref):
        self.env = env
        self.name = name
        self.park = park_ref # Tham chiếu ngược lại Park để ghi log
        
        cap = int(config.get("Nhân viên", 5))
        if config.get("Loại", "Khác") == "Cảnh quan": cap = 9999
            
        self.resource = simpy.Resource(env, capacity=cap)
        self.service_time = config.get("Tốc độ (phút)", 10)
        self.queue_cap = config.get("Sức chứa hàng đợi", 100)
        self.price = config.get("Giá/Chi tiêu (VNĐ)", 0)
        self.failure_rate = config.get("Tỷ lệ hỏng (%)", 0.0)
        
        self.x = config.get("x", random.randint(50, 750))
        self.y = config.get("y", random.randint(50, 550))
        
        # Breakdown Process
        if self.failure_rate > 0:
            self.env.process(self.breakdown_control())

    def breakdown_control(self):
        """Mô phỏng sự cố ngẫu nhiên"""
        while True:
            # Thời gian hoạt động trước khi hỏng
            time_to_fail = random.expovariate(self.failure_rate / 1000.0) if self.failure_rate > 0 else 999999
            yield self.env.timeout(time_to_fail)
            
            # Ghi log sự cố
            self.park.incident_log.append({
                "time": self.env.now,
                "node": self.name,
                "type": "Breakdown",
                "duration": 30 # Giả sử sửa mất 30p
            })
            
            # Chiếm dụng tài nguyên để sửa
            with self.resource.request(priority=-1) as req: # Priority cao để chặn khách
                yield req
                yield self.env.timeout(30)

class DigitalTwinPark:
    def __init__(self, env, nodes_config):
        self.env = env
        self.nodes = {}
        self.node_coords = {} 
        
        # --- LOGGING DATA STORES ---
        self.movement_log = []     # Cho Animation
        self.entry_log = []        # Log khách vào: {time, type}
        self.exit_log = []         # Log khách ra: {time, type}
        self.revenue_log = []      # Log doanh thu: {time, source, amount, category}
        self.snapshot_log = []     # Log trạng thái định kỳ: {time, node, queue_len, status}
        self.incident_log = []     # Log sự cố: {time, node, type}
        
        # --- CẤU HÌNH CỔNG ---
        self.gate_in_pos = (350, 580)
        self.gate_out_pos = (450, 580)
        
        for idx, row in nodes_config.iterrows():
            name = row["Tên Khu"]
            self.nodes[name] = ServiceNode(env, name, row, self)
            self.node_coords[name] = (self.nodes[name].x, self.nodes[name].y)
            
        self.gate_qr = simpy.Resource(env, capacity=4) 
        self.gate_booking = simpy.Resource(env, capacity=2)
        self.gate_walkin = simpy.Resource(env, capacity=2)
        self.current_visitors = 0

    def capture_snapshot(self):
        """Ghi nhận trạng thái hệ thống mỗi 10 phút"""
        while True:
            for name, node in self.nodes.items():
                q_len = len(node.resource.queue)
                status = "Normal"
                if q_len >= node.queue_cap: status = "Overload"
                if node.resource.count == node.resource.capacity and q_len > 0: status = "Busy"
                
                self.snapshot_log.append({
                    "time": self.env.now,
                    "node": name,
                    "queue_len": q_len,
                    "capacity": node.queue_cap,
                    "status": status
                })
            yield self.env.timeout(10) 

def visitor_journey(env, visitor_id, park, is_combo, entry_time):
    v_type = 2 if is_combo else 1
    if "Tour" in visitor_id: v_type = 3
    
    # --- 1. ENTRY LOGIC ---
    current_x, current_y = park.gate_in_pos

    # Check-in Process
    rand_gate = random.random() * 100
    if rand_gate < GATE_QR_PCT:
        with park.gate_qr.request() as req: yield req; yield env.timeout(0.5)
    elif rand_gate < GATE_QR_PCT + GATE_BOOKING_PCT:
        with park.gate_booking.request() as req: yield req; yield env.timeout(2.0)
    else:
        with park.gate_walkin.request() as req: yield req; yield env.timeout(5.0)

    # Update State & Log Entry
    park.current_visitors += 1
    park.entry_log.append({"time": env.now, "type": v_type})
    
    # Log Ticket Revenue
    ticket_rev = TICKET_PRICE_COMBO if is_combo else TICKET_PRICE_ENTRY
    park.revenue_log.append({
        "time": env.now,
        "source": "Cổng Vé",
        "category": "Vé",
        "amount": ticket_rev
    })
    
    # --- 2. PLAY LOGIC ---
    stay_duration = random.gauss(AVG_DWELL_TIME, 30)
    leave_time = entry_time + stay_duration
    node_names = list(park.nodes.keys())
    
    while env.now < leave_time and env.now < TOTAL_MINUTES - 15:
        target_name = random.choice(node_names)
        target_node = park.nodes[target_name]
        
        # Di chuyển (Log Movement)
        travel_time = random.randint(5, 15)
        park.movement_log.append({
            "id": visitor_id, "type": v_type,
            "start": env.now, "end": env.now + travel_time,
            "x1": current_x, "y1": current_y, "x2": target_node.x, "y2": target_node.y
        })
        yield env.timeout(travel_time)
        current_x, current_y = target_node.x, target_node.y
        
        # Sử dụng dịch vụ
        if len(target_node.resource.queue) < target_node.queue_cap:
            arrival_ts = env.now
            with target_node.resource.request() as req:
                yield req 
                yield env.timeout(target_node.service_time)
                
                # Tính tiền dịch vụ (Revenue Log)
                spent = 0
                if target_node.price > 0:
                    if not is_combo: # Vé lẻ phải trả tiền
                        spent = target_node.price
                    elif is_combo and random.random() < 0.2: # Combo thỉnh thoảng mua thêm
                        spent = target_node.price
                
                if spent > 0:
                    park.revenue_log.append({
                        "time": env.now,
                        "source": target_node.name,
                        "category": "Dịch vụ",
                        "amount": spent
                    })
                
            leave_ts = env.now
            # Log đứng yên
            park.movement_log.append({
                "id": visitor_id, "type": v_type,
                "start": arrival_ts, "end": leave_ts,
                "x1": current_x, "y1": current_y, "x2": current_x, "y2": current_y
            })

    # --- 3. EXIT LOGIC ---
    exit_walk_time = random.randint(10, 20)
    park.movement_log.append({
        "id": visitor_id, "type": v_type,
        "start": env.now, "end": env.now + exit_walk_time,
        "x1": current_x, "y1": current_y, "x2": park.gate_out_pos[0], "y2": park.gate_out_pos[1]
    })
    yield env.timeout(exit_walk_time)
    
    park.current_visitors -= 1
    park.exit_log.append({"time": env.now, "type": v_type})

def park_generator(env, park, total_visitors):
    stop_entry_time = TOTAL_MINUTES - STOP_ENTRY_MINUTES
    visitor_count = 0
    interarrival = TOTAL_MINUTES / total_visitors if total_visitors > 0 else 1
    while env.now < stop_entry_time and visitor_count < total_visitors:
        if park.current_visitors < PARK_CAPACITY:
            visitor_count += 1
            is_combo = random.random() < (RATIO_COMBO / 100.0)
            env.process(visitor_journey(env, f"Vis_{visitor_count}", park, is_combo, env.now))
        yield env.timeout(random.expovariate(1.0 / interarrival))

def tour_generator(env, park, tours_df):
    tours = tours_df.to_dict('records')
    for tour in tours: tour['arrival_min'] = time_to_min(tour['Giờ đến'], OPEN_TIME)
    tours.sort(key=lambda x: x['arrival_min'])
    for tour in tours:
        if tour['arrival_min'] > env.now:
            yield env.timeout(tour['arrival_min'] - env.now)
        for i in range(tour['Số lượng']):
            env.process(visitor_journey(env, f"Tour_{tour['Loại đoàn']}_{i}", park, True, env.now))

# ==========================================
# 3. VISUALIZATION COMPONENT (HTML/JS)
# ==========================================
def render_animation(movements, nodes_df, open_hour):
    # (Giữ nguyên phần code HTML/JS render animation như cũ)
    # Rút gọn để tập trung vào phần Analytics, nhưng vẫn chạy tốt
    json_movements = json.dumps(movements)
    nodes_data = nodes_df[["Tên Khu", "x", "y", "Loại"]].to_dict('records')
    json_nodes = json.dumps(nodes_data)

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: sans-serif; background: #0e1117; color: white; margin: 0; }}
            .controls {{ padding: 10px; display: flex; align-items: center; gap: 15px; background: #1f2937; }}
            canvas {{ display: block; background-color: #111827; border-bottom: 2px solid #374151; }}
            .legend-item {{ display: flex; align-items: center; font-size: 12px; margin-right: 10px; }}
            .dot {{ width: 8px; height: 8px; border-radius: 50%; margin-right: 5px; }}
        </style>
    </head>
    <body>
        <div class="controls">
            <div id="clock" style="font-size: 1.2rem; font-weight: bold; width: 60px;">08:00</div>
            <input type="range" id="speed" min="1" max="50" value="10" title="Tốc độ">
            <div style="display:flex;">
                <div class="legend-item"><div class="dot" style="background:#4CAF50"></div>Lẻ</div>
                <div class="legend-item"><div class="dot" style="background:#2196F3"></div>Combo</div>
                <div class="legend-item"><div class="dot" style="background:#FFC107"></div>Đoàn</div>
            </div>
        </div>
        <canvas id="simCanvas" width="800" height="500"></canvas>
        <script>
            const canvas = document.getElementById('simCanvas');
            const ctx = canvas.getContext('2d');
            const movements = {json_movements};
            const nodes = {json_nodes};
            const openHour = {open_hour};
            
            let currentTime = 0;
            const endTime = Math.max(...movements.map(m => m.end));
            let speed = 10;
            document.getElementById('speed').oninput = function() {{ speed = parseInt(this.value); }};

            function minToTime(min) {{
                let h = Math.floor(min / 60) + openHour;
                let m = Math.floor(min % 60);
                return (h < 10 ? '0'+h : h) + ':' + (m < 10 ? '0'+m : m);
            }}

            function draw() {{
                ctx.fillStyle = '#111827'; ctx.fillRect(0, 0, canvas.width, canvas.height);
                
                // Draw Gates
                ctx.fillStyle = '#10B981'; ctx.fillRect(330, 480, 40, 10); ctx.fillText("IN", 340, 475);
                ctx.fillStyle = '#EF4444'; ctx.fillRect(430, 480, 40, 10); ctx.fillText("OUT", 435, 475);

                // Draw Nodes
                nodes.forEach(n => {{
                    ctx.fillStyle = '#374151';
                    if (n['Loại'] === 'Ăn uống') ctx.fillStyle = '#92400e';
                    // Scale Y coordinate to fit 500px height
                    const y = n.y * (500/600); 
                    ctx.fillRect(n.x - 15, y - 15, 30, 30);
                    ctx.fillStyle = '#9ca3af'; ctx.font = '10px sans-serif'; ctx.textAlign = 'center';
                    ctx.fillText(n['Tên Khu'], n.x, y + 25);
                }});
                
                // Draw Agents
                const active = movements.filter(m => currentTime >= m.start && currentTime <= m.end);
                active.forEach(m => {{
                    const duration = m.end - m.start;
                    const progress = duration > 0 ? (currentTime - m.start) / duration : 1;
                    const cx = m.x1 + (m.x2 - m.x1) * progress;
                    // Scale Y
                    const cy = (m.y1 + (m.y2 - m.y1) * progress) * (500/600);
                    
                    if (m.type === 1) ctx.fillStyle = '#4CAF50';
                    else if (m.type === 2) ctx.fillStyle = '#2196F3';
                    else ctx.fillStyle = '#FFC107';
                    
                    ctx.beginPath(); ctx.arc(cx, cy, m.type === 3 ? 4 : 2.5, 0, Math.PI * 2); ctx.fill();
                }});

                document.getElementById('clock').innerText = minToTime(currentTime);
                currentTime += (0.05 * speed);
                if (currentTime < endTime) requestAnimationFrame(draw);
                else {{ currentTime = 0; requestAnimationFrame(draw); }}
            }}
            draw();
        </script>
    </body>
    </html>
    """
    components.html(html_code, height=600)

# ==========================================
# 4. REPORTING & CHARTS FUNCTION
# ==========================================
def generate_report(park, open_time_obj):
    st.markdown("---")
    st.header("📊 BÁO CÁO PHÂN TÍCH VẬN HÀNH (Analytics)")
    
    # --- PREPARE DATA FRAMES ---
    df_entry = pd.DataFrame(park.entry_log)
    df_exit = pd.DataFrame(park.exit_log)
    df_rev = pd.DataFrame(park.revenue_log)
    df_snap = pd.DataFrame(park.snapshot_log)
    
    # Convert Minutes to Hours for Grouping
    if not df_entry.empty: df_entry['Hour'] = df_entry['time'].apply(lambda x: min_to_hour_label(x, open_time_obj))
    if not df_exit.empty: df_exit['Hour'] = df_exit['time'].apply(lambda x: min_to_hour_label(x, open_time_obj))
    if not df_rev.empty: df_rev['Hour'] = df_rev['time'].apply(lambda x: min_to_hour_label(x, open_time_obj))
    if not df_snap.empty: df_snap['Hour'] = df_snap['time'].apply(lambda x: min_to_hour_label(x, open_time_obj))

    # TABS
    tab_flow, tab_rev, tab_ops = st.tabs(["👥 Lưu Lượng (Traffic)", "💰 Doanh Thu (Revenue)", "⚠️ Vận Hành & Sự Cố"])

    # --- TAB 1: TRAFFIC FLOW ---
    with tab_flow:
        st.subheader("1. Phân tích Lưu lượng Khách Vào/Ra")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Group Entry & Exit by Hour
            if not df_entry.empty and not df_exit.empty:
                entry_counts = df_entry.groupby('Hour').size().reset_index(name='Vào (Entry)')
                exit_counts = df_exit.groupby('Hour').size().reset_index(name='Ra (Exit)')
                
                df_traffic = pd.merge(entry_counts, exit_counts, on='Hour', how='outer').fillna(0)
                
                fig_traffic = go.Figure()
                fig_traffic.add_trace(go.Bar(x=df_traffic['Hour'], y=df_traffic['Vào (Entry)'], name='Khách Vào', marker_color='#10B981'))
                fig_traffic.add_trace(go.Bar(x=df_traffic['Hour'], y=df_traffic['Ra (Exit)'], name='Khách Ra', marker_color='#EF4444'))
                fig_traffic.update_layout(title="Lượng khách Vào/Ra theo khung giờ", xaxis_title="Giờ trong ngày", barmode='group')
                st.plotly_chart(fig_traffic, use_container_width=True)
            else:
                st.warning("Chưa có dữ liệu khách.")

        with col2:
            st.write("#### Tổng quan")
            if not df_entry.empty:
                total_in = len(df_entry)
                total_out = len(df_exit)
                st.metric("Tổng lượt vào", f"{total_in:,} khách")
                st.metric("Tổng lượt ra", f"{total_out:,} khách")
                peak_hour = df_traffic.loc[df_traffic['Vào (Entry)'].idxmax()]['Hour'] if not df_traffic.empty else "N/A"
                st.metric("Giờ cao điểm (Vào)", f"{peak_hour}:00")

    # --- TAB 2: REVENUE ANALYSIS ---
    with tab_rev:
        st.subheader("2. Phân tích Doanh thu Chi tiết")
        
        if not df_rev.empty:
            c1, c2 = st.columns(2)
            
            with c1:
                # 2.1 Tỷ trọng doanh thu theo giờ (Vé vs Dịch vụ)
                rev_by_hour_cat = df_rev.groupby(['Hour', 'category'])['amount'].sum().reset_index()
                fig_stack = px.bar(rev_by_hour_cat, x='Hour', y='amount', color='category', 
                                   title="Cơ cấu Doanh thu theo Giờ (Vé vs Dịch vụ)",
                                   labels={'amount': 'Doanh thu (VNĐ)', 'Hour': 'Giờ'},
                                   color_discrete_map={'Vé': '#3B82F6', 'Dịch vụ': '#F59E0B'})
                st.plotly_chart(fig_stack, use_container_width=True)
            
            with c2:
                # 2.2 So sánh doanh thu các quầy dịch vụ (Trừ cổng vé)
                df_service = df_rev[df_rev['category'] == 'Dịch vụ']
                if not df_service.empty:
                    rev_by_node = df_service.groupby('source')['amount'].sum().reset_index().sort_values('amount', ascending=False)
                    fig_node = px.bar(rev_by_node, x='amount', y='source', orientation='h',
                                      title="Top Doanh thu Dịch vụ theo Quầy",
                                      labels={'amount': 'Tổng thu (VNĐ)', 'source': 'Quầy'},
                                      color='amount', color_continuous_scale='Viridis')
                    st.plotly_chart(fig_node, use_container_width=True)
                else:
                    st.info("Chưa phát sinh doanh thu dịch vụ phụ trợ.")

            # 2.3 Heatmap Doanh thu chi tiết theo giờ của các quầy con
            st.write("#### Chi tiết Doanh thu Dịch vụ theo Giờ")
            if not df_service.empty:
                pivot_rev = df_service.pivot_table(index='source', columns='Hour', values='amount', aggfunc='sum').fillna(0)
                fig_heat_rev = px.imshow(pivot_rev, aspect="auto", color_continuous_scale="Greens",
                                        title="Heatmap: Doanh thu theo Giờ & Điểm bán", origin='lower')
                st.plotly_chart(fig_heat_rev, use_container_width=True)
        else:
            st.warning("Chưa có dữ liệu doanh thu.")

    # --- TAB 3: OPERATIONS & INCIDENTS ---
    with tab_ops:
        st.subheader("3. Hiệu quả Vận hành & Sự cố")
        
        c_op1, c_op2 = st.columns(2)
        
        with c_op1:
            # 3.1 Tỷ lệ Quá tải (Overload Rate)
            if not df_snap.empty:
                # Tính % số lần snapshot bị Overload
                overload_stats = df_snap[df_snap['status'] == 'Overload'].groupby('node').size()
                total_stats = df_snap.groupby('node').size()
                pct_overload = (overload_stats / total_stats * 100).fillna(0).reset_index(name='Rate')
                
                fig_overload = px.bar(pct_overload, x='node', y='Rate', 
                                      title="Tỷ lệ Thời gian Quá tải (%)",
                                      labels={'Rate': '% Thời gian Quá tải', 'node': 'Khu vực'},
                                      color='Rate', color_continuous_scale='Reds')
                st.plotly_chart(fig_overload, use_container_width=True)
        
        with c_op2:
            # 3.2 Thống kê sự cố
            if park.incident_log:
                df_inc = pd.DataFrame(park.incident_log)
                inc_counts = df_inc['node'].value_counts().reset_index()
                inc_counts.columns = ['Node', 'Count']
                fig_inc = px.pie(inc_counts, names='Node', values='Count', title="Phân bố Sự cố Hỏng hóc")
                st.plotly_chart(fig_inc, use_container_width=True)
            else:
                st.success("🎉 Tuyệt vời! Không có sự cố kỹ thuật nào xảy ra trong ca vận hành.")
                
        # 3.3 Heatmap Trạng thái (Chi tiết quá tải theo giờ)
        if not df_snap.empty:
            st.write("#### Biểu đồ Nhiệt: Mức độ Đông đúc Trung bình (Khách chờ)")
            # Tính trung bình số người chờ theo giờ
            pivot_q = df_snap.pivot_table(index='node', columns='Hour', values='queue_len', aggfunc='mean').fillna(0)
            fig_q = px.imshow(pivot_q, aspect="auto", color_continuous_scale="OrRd", origin='lower',
                             title="Số lượng khách xếp hàng trung bình theo giờ")
            st.plotly_chart(fig_q, use_container_width=True)

# ==========================================
# 5. RUN
# ==========================================

if st.button("🚀 CHẠY MÔ PHỎNG & PHÂN TÍCH", type="primary"):
    env = simpy.Environment()
    park = DigitalTwinPark(env, edited_nodes_df)
    
    # Processes
    env.process(park_generator(env, park, TOTAL_VISITORS))
    env.process(tour_generator(env, park, edited_tours_df))
    env.process(park.capture_snapshot())
    
    with st.spinner("Đang chạy mô phỏng và thu thập dữ liệu..."):
        env.run(until=TOTAL_MINUTES)
    
    # 1. Show Animation
    st.success("Mô phỏng hoàn tất!")
    render_animation(park.movement_log, edited_nodes_df, OPEN_TIME.hour)
    
    # 2. Show Advanced Analytics Report
    generate_report(park, OPEN_TIME)