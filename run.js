const readline = require('readline');

function exponential(rate) {
  return -Math.log(1 - Math.random()) / rate;
}

function simulateQueue(meanInterarrival, meanService, numCustomers) {
  const arrivalRate = 1 / meanInterarrival;
  const serviceRate = 1 / meanService;

  // Generate arrival times
  let arrivalTimes = [0];
  for (let i = 1; i < numCustomers; i++) {
    const interarrival = exponential(arrivalRate);
    arrivalTimes.push(arrivalTimes[arrivalTimes.length - 1] + interarrival);
  }

  // Generate service times
  let serviceTimes = [];
  for (let i = 0; i < numCustomers; i++) {
    serviceTimes.push(exponential(serviceRate));
  }

  // Compute start and departure times
  let startTimes = new Array(numCustomers).fill(0);
  let departureTimes = new Array(numCustomers).fill(0);

  startTimes[0] = arrivalTimes[0];
  departureTimes[0] = startTimes[0] + serviceTimes[0];

  for (let i = 1; i < numCustomers; i++) {
    startTimes[i] = Math.max(arrivalTimes[i], departureTimes[i - 1]);
    departureTimes[i] = startTimes[i] + serviceTimes[i];
  }

  // Wait times
  let waitTimes = [];
  for (let i = 0; i < numCustomers; i++) {
    waitTimes.push(startTimes[i] - arrivalTimes[i]);
  }
  const avgWaitTime = waitTimes.reduce((a, b) => a + b, 0) / numCustomers || 0;

  // Total time
  const totalTime = departureTimes[departureTimes.length - 1] || 0;

  // Utilization
  const totalServiceTime = serviceTimes.reduce((a, b) => a + b, 0);
  const utilization = totalTime > 0 ? (totalServiceTime / totalTime) * 100 : 0;

  // Avg nis
  let sumTimeInSystem = 0;
  for (let i = 0; i < numCustomers; i++) {
    sumTimeInSystem += departureTimes[i] - arrivalTimes[i];
  }
  const avgNis = totalTime > 0 ? sumTimeInSystem / totalTime : 0;

  // Avg queue length
  const avgQueueLength = totalTime > 0 ? avgNis - (totalServiceTime / totalTime) : 0;

  // Max queue length
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

// Parse command line arguments or use defaults
let meanInterarrival = 3;
let meanService = 5;
let numCustomers = 2000;

const args = process.argv.slice(2);
if (args.length >= 1) meanInterarrival = parseFloat(args[0]);
if (args.length >= 2) meanService = parseFloat(args[1]);
if (args.length >= 3) numCustomers = parseInt(args[2]);

// Run simulation
const results = simulateQueue(meanInterarrival, meanService, numCustomers);
console.log(`Average wait time: ${results.avgWaitTime.toFixed(2)} minutes`);
console.log(`Average queue length: ${results.avgQueueLength.toFixed(2)}`);
console.log(`Max queue length: ${results.maxQueueLength}`);
console.log(`Server utilization: ${results.utilization.toFixed(2)}%`);

// node file.js [interarrival] [service] [customers]