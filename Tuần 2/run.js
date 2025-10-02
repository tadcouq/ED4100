const readline = require('readline');

// Hàm sinh biến ngẫu nhiên phân phối mũ
function exponential(rate) {
  return -Math.log(1 - Math.random()) / rate;
}

// Mô phỏng hệ thống xếp hàng
function simulateQueue(meanInterarrival, meanService, numCustomers) {
  const arrivalRate = 1 / meanInterarrival;
  const serviceRate = 1 / meanService;

  // Sinh thời điểm đến của khách hàng
  let arrivalTimes = [0];
  for (let i = 1; i < numCustomers; i++) {
    const interarrival = exponential(arrivalRate);
    arrivalTimes.push(arrivalTimes[arrivalTimes.length - 1] + interarrival);
  }

  // Sinh thời gian phục vụ cho từng khách hàng
  let serviceTimes = [];
  for (let i = 0; i < numCustomers; i++) {
    serviceTimes.push(exponential(serviceRate));
  }

  // Tính thời điểm bắt đầu phục vụ và thời điểm rời khỏi hệ thống
  let startTimes = new Array(numCustomers).fill(0);
  let departureTimes = new Array(numCustomers).fill(0);

  startTimes[0] = arrivalTimes[0];
  departureTimes[0] = startTimes[0] + serviceTimes[0];

  for (let i = 1; i < numCustomers; i++) {
    startTimes[i] = Math.max(arrivalTimes[i], departureTimes[i - 1]);
    departureTimes[i] = startTimes[i] + serviceTimes[i];
  }

  // Tính thời gian chờ của từng khách hàng
  let waitTimes = [];
  for (let i = 0; i < numCustomers; i++) {
    waitTimes.push(startTimes[i] - arrivalTimes[i]);
  }
  const avgWaitTime = waitTimes.reduce((a, b) => a + b, 0) / numCustomers || 0;

  // Tổng thời gian hoạt động của hệ thống
  const totalTime = departureTimes[departureTimes.length - 1] || 0;

  // Tính mức độ sử dụng của máy chủ
  const totalServiceTime = serviceTimes.reduce((a, b) => a + b, 0);
  const utilization = totalTime > 0 ? (totalServiceTime / totalTime) * 100 : 0;

  // Tính số khách trung bình trong hệ thống
  let sumTimeInSystem = 0;
  for (let i = 0; i < numCustomers; i++) {
    sumTimeInSystem += departureTimes[i] - arrivalTimes[i];
  }
  const avgNis = totalTime > 0 ? sumTimeInSystem / totalTime : 0;

  // Tính độ dài hàng đợi trung bình
  const avgQueueLength = totalTime > 0 ? avgNis - (totalServiceTime / totalTime) : 0;

  // Tính độ dài hàng đợi lớn nhất
  let events = [];
  for (let i = 0; i < numCustomers; i++) {
    events.push({time: arrivalTimes[i], type: 'a'});
    events.push({time: departureTimes[i], type: 'd'});
  }

  events.sort((a, b) => a.time - b.time);

  let currentNis = 0;
  let maxNis = 0;
  for (let event of events) {
    if (event.type === 'a') {
      currentNis += 1;
      maxNis = Math.max(maxNis, currentNis);
    } else {
      currentNis -= 1;
    }
  }

  const maxQueueLength = Math.max(0, maxNis - 1);

  return {
    avgWaitTime,
    avgQueueLength,
    maxQueueLength,
    utilization
  };
}

// Đọc tham số từ dòng lệnh hoặc dùng giá trị mặc định
let meanInterarrival = 3;
let meanService = 5;
let numCustomers = 2000;

const args = process.argv.slice(2);
if (args.length >= 1) meanInterarrival = parseFloat(args[0]);
if (args.length >= 2) meanService = parseFloat(args[1]);
if (args.length >= 3) numCustomers = parseInt(args[2]);

// Chạy mô phỏng
const results = simulateQueue(meanInterarrival, meanService, numCustomers);
console.log(`Thời gian chờ trung bình: ${results.avgWaitTime.toFixed(2)} phút`);
console.log(`Độ dài hàng đợi trung bình: ${results.avgQueueLength.toFixed(2)}`);
console.log(`Độ dài hàng đợi lớn nhất: ${results.maxQueueLength}`);
console.log(`Mức độ sử dụng tối ưu nhân sự: ${results.utilization.toFixed(2)}%`);

// node run.js [interarrival] [service] [customers]