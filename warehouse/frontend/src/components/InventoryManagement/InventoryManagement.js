import React, { useState, useEffect } from 'react';
import axios from 'axios';
import SummaryCards from './SummaryCards';
import './InventoryManagement.css';

const InventoryManagement = () => {
  const [inventory, setInventory] = useState([]);
  const [formData, setFormData] = useState({
    product_name: '',
    selling_price: '',
    date_of_entry: '',
    date_of_exit: '',
    date_of_expiry: '',
    quantity: '',
    cost_price: '',
    storage_slot: '',
  });

  const [matrix, setMatrix] = useState(Array(5).fill(Array(30).fill('white')));
  const [selectedSlot, setSelectedSlot] = useState(null);

  useEffect(() => {
    const fetchInventory = async () => {
      try {
        const response = await axios.get('http://localhost:5000/api/inventory');
        setInventory(response.data);
      } catch (error) {
        console.error('Error fetching inventory:', error);
      }
    };
    fetchInventory();
  }, []);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData({ ...formData, [name]: value });
  };

  const handleBoxClick = (row, col) => {
    setSelectedSlot(`${row}-${col}`);
    setMatrix(
      matrix.map((r, rowIndex) =>
        r.map((c, colIndex) =>
          rowIndex === row && colIndex === col ? 'orange' : c === 'green' ? 'green' : 'white'
        )
      )
    );
  };

  const handleFormSubmit = async (e) => {
    e.preventDefault();
    if (!selectedSlot) {
      alert("Please select a storage slot.");
      return;
    }
  
    try {
      const payload = {
        ...formData,
        storage_slot: selectedSlot,
      };
  
      console.log("Submitting payload:", payload); 
  
      const response = await axios.post("http://localhost:5000/api/inventory", payload);
  
      console.log("Response from server:", response.data);
      alert("Product added successfully!");
  
      window.location.reload();
    } catch (error) {
      console.error("Error submitting form:", error);
    }
  };
  

  return (
    <div className="inventory-management">
      <h1>Inventory Management</h1>

      <SummaryCards inventory={inventory} />

      <form className="inventory-form" onSubmit={handleFormSubmit}>
        <input
          type="text"
          name="product_name"
          value={formData.product_name}
          placeholder="Product Name"
          onChange={handleInputChange}
          required
        />
        <input
          type="number"
          name="selling_price"
          value={formData.selling_price}
          placeholder="Selling Price"
          onChange={handleInputChange}
          required
        />
        <div className="labeled-input">
          <label>Entry</label>
          <input
            type="date"
            name="date_of_entry"
            value={formData.date_of_entry}
            onChange={handleInputChange}
            required
          />
        </div>
        <div className="labeled-input">
          <label>Exit</label>
          <input
            type="date"
            name="date_of_exit"
            value={formData.date_of_exit}
            onChange={handleInputChange}
            required
          />
        </div>
        <div className="labeled-input">
          <label>Expiry</label>
          <input
            type="date"
            name="date_of_expiry"
            value={formData.date_of_expiry}
            onChange={handleInputChange}
            required
          />
        </div>
        <input
          type="number"
          name="quantity"
          value={formData.quantity}
          placeholder="Quantity"
          onChange={handleInputChange}
          required
        />
        <input
          type="number"
          name="cost_price"
          value={formData.cost_price}
          placeholder="Cost Price"
          onChange={handleInputChange}
          required
        />
        <button type="submit">Add Product</button>
      </form>

      <div className="box-matrix">
        {matrix.map((row, rowIndex) => (
          <div key={rowIndex} className="matrix-row">
            {row.map((color, colIndex) => (
              <div
                key={`${rowIndex}-${colIndex}`}
                className="matrix-box"
                style={{ backgroundColor: color }}
                onClick={() => handleBoxClick(rowIndex, colIndex)}
              ></div>
            ))}
          </div>
        ))}
      </div>

      <table className="inventory-table">
        <thead>
          <tr>
            <th>Name</th>
            <th>Entry</th>
            <th>Exit</th>
            <th>Expiry</th>
            <th>Stock</th>
            <th>Cost Price</th>
            <th>Selling Price</th>
            <th>Storage Slot</th>
          </tr>
        </thead>
        <tbody>
          {inventory.map((item) => (
            <tr key={item.id}>
              <td>{item.product_name}</td>
              <td>{item.date_of_entry}</td>
              <td>{item.date_of_exit}</td>
              <td>{item.date_of_expiry}</td>
              <td>{item.quantity}</td>
              <td>{item.cost_price}</td>
              <td>{item.selling_price}</td>
              <td>{item.storage_slot}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default InventoryManagement;