import random
import streamlit as st

def simulate_queue(mean_interarrival, mean_service, num_customers):
    # Arrival rate and service rate
    arrival_rate = 1 / mean_interarrival
    service_rate = 1 / mean_service
    
    # Generate arrival times (cumulative)
    arrival_times = [0]
    for i in range(1, num_customers):
        interarrival = random.expovariate(arrival_rate)
        arrival_times.append(arrival_times[-1] + interarrival)
    
    # Generate service times
    service_times = [random.expovariate(service_rate) for _ in range(num_customers)]
    
    # Compute start and departure times
    start_times = [0] * num_customers
    departure_times = [0] * num_customers
    
    start_times[0] = arrival_times[0]
    departure_times[0] = start_times[0] + service_times[0]
    
    for i in range(1, num_customers):
        start_times[i] = max(arrival_times[i], departure_times[i-1])
        departure_times[i] = start_times[i] + service_times[i]
    
    # Wait times
    wait_times = [start_times[i] - arrival_times[i] for i in range(num_customers)]
    avg_wait_time = sum(wait_times) / num_customers if num_customers > 0 else 0
    
    # Total simulation time (end at last departure)
    total_time = departure_times[-1] if departure_times else 0
    
    # Utilization
    total_service_time = sum(service_times)
    utilization = (total_service_time / total_time) * 100 if total_time > 0 else 0
    
    # Average number in system (L = sum time in system / total time)
    sum_time_in_system = sum(departure_times[i] - arrival_times[i] for i in range(num_customers))
    avg_nis = sum_time_in_system / total_time if total_time > 0 else 0
    
    # Average queue length (Lq = L - rho, where rho = utilization / 100)
    avg_queue_length = avg_nis - (total_service_time / total_time) if total_time > 0 else 0
    
    # Max queue length (max nis - 1)
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

# Streamlit app
st.title("Bank Queue Simulation")

# Inputs with defaults
mean_interarrival = st.number_input("Mean interarrival time (minutes)", value=3.0)
mean_service = st.number_input("Mean service time (minutes)", value=5.0)
num_customers = st.number_input("Number of customers", value=2000)

if st.button("Run Simulation"):
    results = simulate_queue(mean_interarrival, mean_service, num_customers)
    st.write(f"Average wait time: {results['avg_wait_time']:.2f} minutes")
    st.write(f"Average queue length: {results['avg_queue_length']:.2f}")
    st.write(f"Max queue length: {results['max_queue_length']}")
    st.write(f"Server utilization: {results['utilization']:.2f}%")