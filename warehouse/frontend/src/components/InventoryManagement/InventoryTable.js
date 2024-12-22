import React from 'react';
import './InventoryTable.css';

const InventoryTable = ({ inventory, onDeleteProduct, onEditProduct }) => {
  return (
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
          <th>Actions</th>
        </tr>
      </thead>
      <tbody>
        {inventory.map((item) => (
          <tr key={item.id}>
            <td>{item.name}</td>
            <td>{item.entryDate}</td>
            <td>{item.exitDate || 'N/A'}</td>
            <td>{item.expiryDate || 'N/A'}</td>
            <td>{item.quantity}</td>
            <td>{item.costPrice}</td>
            <td>{item.sellingPrice}</td>
            <td>
              <button onClick={() => onEditProduct(item.id)}>Edit</button>
              <button onClick={() => onDeleteProduct(item.id)}>Delete</button>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
};

export default InventoryTable;