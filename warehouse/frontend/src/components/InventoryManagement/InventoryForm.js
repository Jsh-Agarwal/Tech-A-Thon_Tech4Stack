import React, { useState } from 'react';

const InventoryForm = ({ onAddProduct }) => {
  const [formData, setFormData] = useState({
    name: '',
    sellingPrice: '',
    entryDate: '',
    exitDate: '',
    expiryDate: '',
    quantity: '',
    costPrice: '',
  });

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    onAddProduct(formData);
    setFormData({
      name: '',
      sellingPrice: '',
      entryDate: '',
      exitDate: '',
      expiryDate: '',
      quantity: '',
      costPrice: '',
    });
  };

  return (
    <form className="inventory-form" onSubmit={handleSubmit}>
      <input
        type="text"
        name="name"
        placeholder="Add Product Name"
        value={formData.name}
        onChange={handleChange}
        required
      />
      <input
        type="number"
        name="sellingPrice"
        placeholder="Add Selling Price"
        value={formData.sellingPrice}
        onChange={handleChange}
        required
      />
      <input
        type="date"
        name="entryDate"
        value={formData.entryDate}
        onChange={handleChange}
        required
      />
      <input
        type="date"
        name="exitDate"
        value={formData.exitDate}
        onChange={handleChange}
      />
      <input
        type="date"
        name="expiryDate"
        value={formData.expiryDate}
        onChange={handleChange}
      />
      <input
        type="number"
        name="quantity"
        placeholder="Add Quantity"
        value={formData.quantity}
        onChange={handleChange}
        required
      />
      <input
        type="number"
        name="costPrice"
        placeholder="Add Cost Price"
        value={formData.costPrice}
        onChange={handleChange}
        required
      />
      <button type="submit">Create Product</button>
    </form>
  );
};

export default InventoryForm;