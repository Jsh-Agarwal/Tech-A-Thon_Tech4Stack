import React, { useState } from 'react';
import './TaskList.css';

const TaskList = ({ tasks, onStatusChange }) => {
  const [selectedTask, setSelectedTask] = useState(null);

  const openModal = (task) => {
    setSelectedTask(task);
  };

  const closeModal = () => {
    setSelectedTask(null);
  };

  return (
    <div className="task-list">
      <ul>
        {tasks.map((task) => (
          <li key={task.id} className={`task-item priority-${task.priority.toLowerCase()}`}>
            <h3 onClick={() => openModal(task)}>{task.title}</h3>
            <p>{task.description}</p>
            <div className="task-actions">
              <label>Status:</label>
              <select
                value={task.status}
                onChange={(e) => onStatusChange(task.id, e.target.value)}
              >
                <option value="Pending">Pending</option>
                <option value="In Progress">In Progress</option>
                <option value="Completed">Completed</option>
              </select>
            </div>
          </li>
        ))}
      </ul>

      {selectedTask && (
  <div
    className="modal"
    onClick={(e) => {
      if (e.target.className === 'modal') closeModal();
    }}
  >
    <div className="modal-content">
      <span className="close-btn" onClick={closeModal}>
        &times;
      </span>
      <h2>{selectedTask.title}</h2>
      <p><strong>Description:</strong> {selectedTask.description}</p>
      <p><strong>Priority:</strong> {selectedTask.priority}</p>
      <p><strong>Status:</strong> {selectedTask.status}</p>
    </div>
  </div>
)}
    </div>
  );
};

export default TaskList;