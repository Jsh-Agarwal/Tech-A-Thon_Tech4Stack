import { faker } from '@faker-js/faker';

export const generateSummaryData = () => [
  { title: 'Total Boxes', value: faker.number.int({ min: 100, max: 500 }), icon: '📦' },
  { title: 'Tasks Completed', value: faker.number.int({ min: 50, max: 300 }), icon: '✅' },
  { title: 'Path Efficiency', value: `${faker.number.int({ min: 85, max: 99 })}%`, icon: '🚀' },
  { title: 'Warehouse Staff', value: faker.number.int({ min: 10, max: 30 }), icon: '👷' },
];

export const generateMetricsData = (range = 'week') => {
  const data = [];
  const rangeDays = range === 'day' ? 7 : range === 'week' ? 4 : 12;
  for (let i = 0; i < rangeDays; i++) {
    data.push({
      name: range === 'day' ? `Day ${i + 1}` : range === 'week' ? `Week ${i + 1}` : `Month ${i + 1}`,
      tasks: faker.number.int({ min: 10, max: 50 }),
      boxes: faker.number.int({ min: 20, max: 100 }),
    });
  }
  return data;
};