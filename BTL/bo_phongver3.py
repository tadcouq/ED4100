import streamlit as st
import simpy
import random
import pandas as pd
import numpy as np
import plotly.express as px
import json
import streamlit.components.v1 as components
from datetime import datetime, timedelta, time

# ==========================================
# 0. CẤU HÌNH & HÀM TIỆN ÍCH
# ==========================================
st.set_page_config(page_title="Digital Twin Park V6 (Exit Flow)", layout="wide")

def time_to_min(time_obj, start_time_obj):
    delta = datetime.combine(datetime.today(), time_obj) - datetime.combine(datetime.today(), start_time_obj)
    return int(delta.total_seconds() / 60)

def min_to_time_str(minutes, start_time_obj):
    new_time = datetime.combine(datetime.today(), start_time_obj) + timedelta(minutes=minutes)
    return new_time.strftime("%H:%M")

# ==========================================
# 1. INPUT MODULE
# ==========================================
st.title("🔥 Digital Twin V6: Full Loop Simulation (In -> Play -> Out)")
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
    TOTAL_VISITORS = st.number_input("Tổng khách dự kiến", value=500) 

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
    st.info("💡 Cổng Vào tại (350, 580) - Cổng Ra tại (450, 580). Hãy bố trí các khu vực hợp lý.")
    default_nodes = [
        {"Tên Khu": "Tàu lượn", "Loại": "Trò chơi", "Nhân viên": 3, "Tốc độ (phút)": 5, "Sức chứa hàng đợi": 50, "Giá/Chi tiêu (VNĐ)": 50000, "x": 100, "y": 100},
        {"Tên Khu": "Nhà hàng", "Loại": "Ăn uống", "Nhân viên": 5, "Tốc độ (phút)": 30, "Sức chứa hàng đợi": 100, "Giá/Chi tiêu (VNĐ)": 150000, "x": 400, "y": 300},
        {"Tên Khu": "Đu quay", "Loại": "Trò chơi", "Nhân viên": 2, "Tốc độ (phút)": 8, "Sức chứa hàng đợi": 30, "Giá/Chi tiêu (VNĐ)": 30000, "x": 700, "y": 100},
        {"Tên Khu": "Quảng trường", "Loại": "Cảnh quan", "Nhân viên": 100, "Tốc độ (phút)": 15, "Sức chứa hàng đợi": 1000, "Giá/Chi tiêu (VNĐ)": 0, "x": 400, "y": 500},
        {"Tên Khu": "Vườn hoa", "Loại": "Cảnh quan", "Nhân viên": 100, "Tốc độ (phút)": 20, "Sức chứa hàng đợi": 1000, "Giá/Chi tiêu (VNĐ)": 0, "x": 700, "y": 500},
        {"Tên Khu": "Toilet A", "Loại": "Tiện ích", "Nhân viên": 10, "Tốc độ (phút)": 5, "Sức chứa hàng đợi": 20, "Giá/Chi tiêu (VNĐ)": 0, "x": 100, "y": 500},
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
    def __init__(self, env, name, config):
        self.env = env
        self.name = name
        
        cap = int(config.get("Nhân viên", 5))
        if config.get("Loại", "Khác") == "Cảnh quan": cap = 9999
            
        self.resource = simpy.Resource(env, capacity=cap)
        self.service_time = config.get("Tốc độ (phút)", 10)
        self.queue_cap = config.get("Sức chứa hàng đợi", 100)
        self.price = config.get("Giá/Chi tiêu (VNĐ)", 0)
        self.rebuy_prob = 0.1 
        
        self.revenue = 0
        self.visits = 0
        
        self.x = config.get("x", random.randint(50, 750))
        self.y = config.get("y", random.randint(50, 550))

class DigitalTwinPark:
    def __init__(self, env, nodes_config):
        self.env = env
        self.nodes = {}
        self.node_coords = {} 
        self.movement_log = [] 
        
        # --- CẤU HÌNH CỔNG ---
        self.gate_in_pos = (350, 580)  # Cổng vào (Trái)
        self.gate_out_pos = (450, 580) # Cổng ra (Phải)
        
        for idx, row in nodes_config.iterrows():
            name = row["Tên Khu"]
            self.nodes[name] = ServiceNode(env, name, row)
            self.node_coords[name] = (self.nodes[name].x, self.nodes[name].y)
            
        self.gate_qr = simpy.Resource(env, capacity=4) 
        self.gate_booking = simpy.Resource(env, capacity=2)
        self.gate_walkin = simpy.Resource(env, capacity=2)

        self.gate_revenue = 0
        self.current_visitors = 0
        self.snapshots = [] 

    def capture_snapshot(self):
        while True:
            snapshot_time = min_to_time_str(self.env.now, OPEN_TIME)
            for name, node in self.nodes.items():
                count = len(node.resource.queue) + node.resource.count
                self.snapshots.append({"Time": snapshot_time, "Node": name, "Visitors": count})
            yield self.env.timeout(30) 

def visitor_journey(env, visitor_id, park, is_combo, entry_time):
    v_type = 2 if is_combo else 1
    if "Tour" in visitor_id: v_type = 3
    
    # 1. BẮT ĐẦU TẠI CỔNG VÀO
    current_x, current_y = park.gate_in_pos

    # Check-in
    rand_gate = random.random() * 100
    if rand_gate < GATE_QR_PCT:
        with park.gate_qr.request() as req: yield req; yield env.timeout(0.5)
    elif rand_gate < GATE_QR_PCT + GATE_BOOKING_PCT:
        with park.gate_booking.request() as req: yield req; yield env.timeout(2.0)
    else:
        with park.gate_walkin.request() as req: yield req; yield env.timeout(5.0)

    park.current_visitors += 1
    if is_combo: park.gate_revenue += TICKET_PRICE_COMBO
    else: park.gate_revenue += TICKET_PRICE_ENTRY
    
    # 2. QUÁ TRÌNH VUI CHƠI
    stay_duration = random.gauss(AVG_DWELL_TIME, 30)
    leave_time = entry_time + stay_duration # Thời điểm quyết định đi về
    
    node_names = list(park.nodes.keys())
    
    # Vòng lặp: còn thời gian & công viên chưa đóng cửa
    while env.now < leave_time and env.now < TOTAL_MINUTES - 15: # Trừ 15p để kịp đi ra
        target_name = random.choice(node_names)
        target_node = park.nodes[target_name]
        dest_x, dest_y = target_node.x, target_node.y
        
        # Di chuyển đến trạm
        travel_time = random.randint(5, 15)
        park.movement_log.append({
            "id": visitor_id, "type": v_type,
            "start": env.now, "end": env.now + travel_time,
            "x1": current_x, "y1": current_y,
            "x2": dest_x, "y2": dest_y
        })
        yield env.timeout(travel_time)
        current_x, current_y = dest_x, dest_y
        
        # Tại trạm (Xếp hàng + Chơi)
        if len(target_node.resource.queue) < target_node.queue_cap:
            arrival_ts = env.now
            with target_node.resource.request() as req:
                yield req 
                yield env.timeout(target_node.service_time)
                
                if target_node.price > 0 and not is_combo:
                    target_node.revenue += target_node.price
                elif target_node.price > 0 and is_combo and random.random() < target_node.rebuy_prob:
                     target_node.revenue += target_node.price
                target_node.visits += 1
                
            leave_ts = env.now
            # Ghi log đứng yên
            park.movement_log.append({
                "id": visitor_id, "type": v_type,
                "start": arrival_ts, "end": leave_ts,
                "x1": current_x, "y1": current_y,
                "x2": current_x, "y2": current_y
            })

    # 3. LUỒNG CỬA RA (EXIT FLOW) - MỚI
    # Khi hết giờ chơi, đi từ vị trí hiện tại -> Cổng Ra
    exit_walk_time = random.randint(10, 20)
    
    park.movement_log.append({
        "id": visitor_id, "type": v_type,
        "start": env.now, "end": env.now + exit_walk_time,
        "x1": current_x, "y1": current_y,
        "x2": park.gate_out_pos[0], "y2": park.gate_out_pos[1] # Đích đến là Cổng Ra
    })
    
    yield env.timeout(exit_walk_time)
    park.current_visitors -= 1

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
    json_movements = json.dumps(movements)
    nodes_data = nodes_df[["Tên Khu", "x", "y", "Loại"]].to_dict('records')
    json_nodes = json.dumps(nodes_data)

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: sans-serif; background: #0e1117; color: white; }}
            .controls {{ margin-bottom: 10px; display: flex; align-items: center; gap: 15px; }}
            canvas {{ background-color: #1f2937; border-radius: 8px; border: 1px solid #374151; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }}
            .legend-item {{ display: flex; align-items: center; font-size: 12px; margin-right: 10px; }}
            .dot {{ width: 8px; height: 8px; border-radius: 50%; margin-right: 5px; }}
            input[type=range] {{ accent-color: #ff4b4b; }}
        </style>
    </head>
    <body>
        <div class="controls">
            <div id="clock" style="font-size: 1.5rem; font-weight: bold; width: 80px;">08:00</div>
            <div>
                <label>Tốc độ:</label>
                <input type="range" id="speed" min="1" max="50" value="10">
            </div>
            <div style="display:flex;">
                <div class="legend-item"><div class="dot" style="background:#4CAF50"></div>Lẻ</div>
                <div class="legend-item"><div class="dot" style="background:#2196F3"></div>Combo</div>
                <div class="legend-item"><div class="dot" style="background:#FFC107"></div>Đoàn</div>
            </div>
        </div>
        
        <canvas id="simCanvas" width="800" height="600"></canvas>

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
                // 1. Clear
                ctx.fillStyle = '#1f2937';
                ctx.fillRect(0, 0, canvas.width, canvas.height);
                
                // 2. Draw Infrastructure
                
                // Cổng Vào (Xanh)
                ctx.fillStyle = '#10B981';
                ctx.fillRect(330, 580, 40, 10);
                ctx.fillText("CỔNG VÀO", 330, 570);
                
                // Cổng Ra (Đỏ)
                ctx.fillStyle = '#EF4444';
                ctx.fillRect(430, 580, 40, 10);
                ctx.fillText("CỔNG RA", 430, 570);

                // Các node
                nodes.forEach(n => {{
                    ctx.fillStyle = '#374151';
                    if (n['Loại'] === 'Cảnh quan') ctx.fillStyle = '#166534';
                    if (n['Loại'] === 'Ăn uống') ctx.fillStyle = '#92400e';
                    
                    ctx.fillRect(n.x - 20, n.y - 20, 40, 40);
                    ctx.fillStyle = '#d1d5db';
                    ctx.font = '10px sans-serif';
                    ctx.textAlign = 'center';
                    ctx.fillText(n['Tên Khu'], n.x, n.y + 30);
                }});
                
                // 3. Draw Agents
                const active = movements.filter(m => currentTime >= m.start && currentTime <= m.end);
                
                active.forEach(m => {{
                    const duration = m.end - m.start;
                    const progress = duration > 0 ? (currentTime - m.start) / duration : 1;
                    
                    const cx = m.x1 + (m.x2 - m.x1) * progress;
                    const cy = m.y1 + (m.y2 - m.y1) * progress;
                    
                    if (m.type === 1) ctx.fillStyle = '#4CAF50';
                    else if (m.type === 2) ctx.fillStyle = '#2196F3';
                    else ctx.fillStyle = '#FFC107';
                    
                    ctx.beginPath();
                    ctx.arc(cx, cy, m.type === 3 ? 4 : 2.5, 0, Math.PI * 2);
                    ctx.fill();
                }});

                // 4. Loop
                document.getElementById('clock').innerText = minToTime(currentTime);
                currentTime += (0.05 * speed);
                if (currentTime < endTime) {{
                    requestAnimationFrame(draw);
                }} else {{
                    currentTime = 0;
                    requestAnimationFrame(draw);
                }}
            }}
            
            draw();
        </script>
    </body>
    </html>
    """
    components.html(html_code, height=700)

# ==========================================
# 4. RUN
# ==========================================

if st.button("🚀 CHẠY MÔ PHỎNG", type="primary"):
    env = simpy.Environment()
    park = DigitalTwinPark(env, edited_nodes_df)
    
    env.process(park_generator(env, park, TOTAL_VISITORS))
    env.process(tour_generator(env, park, edited_tours_df))
    env.process(park.capture_snapshot())
    
    with st.spinner("Đang tính toán logic SimPy & Xây dựng Animation..."):
        env.run(until=TOTAL_MINUTES)
    
    st.markdown("### 🎥 Mô phỏng Trực quan")
    st.caption("Khách sẽ đi từ Cổng Vào -> Các Node -> Cổng Ra (khi hết giờ)")
    render_animation(park.movement_log, edited_nodes_df, OPEN_TIME.hour)
    
    # Analytics Tabs
    tab_map, tab_fin = st.tabs(["🔥 Heatmap", "💰 Doanh thu"])
    
    with tab_map:
        if park.snapshots:
            df_snap = pd.DataFrame(park.snapshots)
            df_pivot = df_snap.pivot_table(index="Node", columns="Time", values="Visitors", aggfunc='sum').fillna(0)
            fig_heat = px.imshow(df_pivot, aspect="auto", color_continuous_scale="RdYlGn_r", origin='lower')
            st.plotly_chart(fig_heat, use_container_width=True)

    with tab_fin:
        service_revenue = sum([n.revenue for n in park.nodes.values()])
        total_rev = park.gate_revenue + service_revenue
        st.metric("Tổng Doanh Thu", f"{total_rev:,.0f} VNĐ")