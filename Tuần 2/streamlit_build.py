import random
import streamlit as st

def simulate_queue(mean_interarrival, mean_service, num_customers):
    # Tỷ lệ đến và tỷ lệ phục vụ
    arrival_rate = 1 / mean_interarrival
    service_rate = 1 / mean_service
    
    # Sinh thời gian đến (cộng dồn)
    arrival_times = [0]
    for i in range(1, num_customers):
        interarrival = random.expovariate(arrival_rate)
        arrival_times.append(arrival_times[-1] + interarrival)
    
    # Sinh thời gian phục vụ
    service_times = [random.expovariate(service_rate) for _ in range(num_customers)]
    
    # Tính thời gian bắt đầu và thời gian rời đi
    start_times = [0] * num_customers
    departure_times = [0] * num_customers
    
    start_times[0] = arrival_times[0]
    departure_times[0] = start_times[0] + service_times[0]
    
    for i in range(1, num_customers):
        start_times[i] = max(arrival_times[i], departure_times[i-1])
        departure_times[i] = start_times[i] + service_times[i]
    
    # Thời gian chờ
    wait_times = [start_times[i] - arrival_times[i] for i in range(num_customers)]
    avg_wait_time = sum(wait_times) / num_customers if num_customers > 0 else 0
    
    # Tổng thời gian mô phỏng (kết thúc tại lần rời đi cuối cùng)
    total_time = departure_times[-1] if departure_times else 0
    
    # Mức độ sử dụng
    total_service_time = sum(service_times)
    utilization = (total_service_time / total_time) * 100 if total_time > 0 else 0
    
    # Số khách trung bình trong hệ thống (L = tổng thời gian trong hệ / tổng thời gian)
    sum_time_in_system = sum(departure_times[i] - arrival_times[i] for i in range(num_customers))
    avg_nis = sum_time_in_system / total_time if total_time > 0 else 0
    
    # Độ dài hàng đợi trung bình (Lq = L - rho, với rho = mức độ sử dụng / 100)
    avg_queue_length = avg_nis - (total_service_time / total_time) if total_time > 0 else 0
    
    # Độ dài hàng đợi lớn nhất (max nis - 1)
    events = []
    for i in range(num_customers):
        events.append((arrival_times[i], 'a'))
        events.append((departure_times[i], 'd'))
    
    events.sort(key=lambda x: x[0])
    
    current_nis = 0
    max_nis = 0
    for time, event_type in events:
        if event_type == 'a':
            current_nis += 1
            max_nis = max(max_nis, current_nis)
        else:
            current_nis -= 1
    
    max_queue_length = max(0, max_nis - 1)
    
    return {
        "avg_wait_time": avg_wait_time,
        "avg_queue_length": avg_queue_length,
        "max_queue_length": max_queue_length,
        "utilization": utilization
    }

# Ứng dụng Streamlit
st.title("Mô phỏng hàng đợi tại ngân hàng")

# Nhập liệu với giá trị mặc định
mean_interarrival = st.number_input("Thời gian đến trung bình (phút)", value=3.0)
mean_service = st.number_input("Thời gian phục vụ trung bình (phút)", value=5.0)
num_customers = st.number_input("Số lượng khách hàng", value=2000)

if st.button("Chạy mô phỏng"):
    results = simulate_queue(mean_interarrival, mean_service, num_customers)
    st.write(f"Thời gian chờ trung bình: {results['avg_wait_time']:.2f} phút")
    st.write(f"Độ dài hàng đợi trung bình: {results['avg_queue_length']:.2f}")
    st.write(f"Độ dài hàng đợi lớn nhất: {results['max_queue_length']}")
    st.write(f"Mức độ tối ưu nhân sự: {results['utilization']:.2f}%")