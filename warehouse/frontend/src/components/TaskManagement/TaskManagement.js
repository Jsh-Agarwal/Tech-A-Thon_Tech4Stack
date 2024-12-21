import React, { useState } from 'react';
import TaskList from './TaskList';
import TaskForm from './TaskForm';
import { sampleTasks } from './taskData';
import './TaskManagement.css';

const TaskManagement = () => {
  const [tasks, setTasks] = useState(sampleTasks);
  const [priorityFilter, setPriorityFilter] = useState('All');
  const [statusFilter, setStatusFilter] = useState('All');
  const [sortCriteria, setSortCriteria] = useState('None');

  const addTask = (newTask) => {
    setTasks([...tasks, { ...newTask, id: tasks.length + 1 }]);
  };

  const updateTaskStatus = (id, status) => {
    setTasks(
      tasks.map((task) => (task.id === id ? { ...task, status } : task))
    );
  };

  const handleSort = (criteria) => {
    setSortCriteria(criteria);
  };

  const filteredTasks = tasks.filter((task) => {
    const matchesPriority = priorityFilter === 'All' || task.priority === priorityFilter;
    const matchesStatus = statusFilter === 'All' || task.status === statusFilter;
    return matchesPriority && matchesStatus;
  });

  const sortedTasks = [...filteredTasks].sort((a, b) => {
    if (sortCriteria === 'Title') {
      return a.title.localeCompare(b.title);
    } else if (sortCriteria === 'Priority') {
      const priorityOrder = { High: 1, Medium: 2, Low: 3 };
      return priorityOrder[a.priority] - priorityOrder[b.priority];
    } else if (sortCriteria === 'Status') {
      const statusOrder = { Pending: 1, 'In Progress': 2, Completed: 3 };
      return statusOrder[a.status] - statusOrder[b.status];
    }
    return 0;
  });

  return (
    <div className="task-management">
      <h1>Task Management</h1>
      <div className="filters">
        <select value={priorityFilter} onChange={(e) => setPriorityFilter(e.target.value)}>
          <option value="All">All Priorities</option>
          <option value="High">High</option>
          <option value="Medium">Medium</option>
          <option value="Low">Low</option>
        </select>
        <select value={statusFilter} onChange={(e) => setStatusFilter(e.target.value)}>
          <option value="All">All Statuses</option>
          <option value="Pending">Pending</option>
          <option value="In Progress">In Progress</option>
          <option value="Completed">Completed</option>
        </select>
        <select value={sortCriteria} onChange={(e) => handleSort(e.target.value)}>
          <option value="None">Sort By</option>
          <option value="Title">Title</option>
          <option value="Priority">Priority</option>
          <option value="Status">Status</option>
        </select>
      </div>
      <TaskForm onAddTask={addTask} />
      <TaskList tasks={sortedTasks} onStatusChange={updateTaskStatus} />
    </div>
  );
};

export default TaskManagement;