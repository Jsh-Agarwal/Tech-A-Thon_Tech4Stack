import { faker } from '@faker-js/faker';

export const sampleTasks = [
  {
    id: 1,
    title: 'Inspect storage area A',
    description: 'Check for damaged boxes.',
    priority: 'High',
    status: 'Pending',
  },
  {
    id: 2,
    title: 'Organize section B',
    description: 'Rearrange boxes to optimize space.',
    priority: 'Medium',
    status: 'In Progress',
  },
  {
    id: 3,
    title: 'Perform safety drill',
    description: 'Conduct a safety drill for warehouse staff.',
    priority: 'Low',
    status: 'Completed',
  },
];

const generateFakeTasks = (count) => {
  const priorities = ['High', 'Medium', 'Low'];
  const statuses = ['Pending', 'In Progress', 'Completed'];

  return Array.from({ length: count }, (_, index) => ({
    id: sampleTasks.length + index + 1,
    title: faker.company.catchPhrase(),
    description: faker.lorem.sentence(),
    priority: faker.helpers.arrayElement(priorities),
    status: faker.helpers.arrayElement(statuses),
  }));
};

const additionalTasks = generateFakeTasks(10);
export const allTasks = [...sampleTasks, ...additionalTasks];