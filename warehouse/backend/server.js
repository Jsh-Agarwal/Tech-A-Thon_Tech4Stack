const express = require("express");
const sqlite3 = require("sqlite3").verbose();
const cors = require("cors");
const bodyParser = require("body-parser");

const app = express();
const PORT = 5000;

app.use(cors());
app.use(bodyParser.json());

// Connect to SQLite
const db = new sqlite3.Database("./database.db", (err) => {
  if (err) {
    console.error("Error connecting to database:", err.message);
  } else {
    console.log("Connected to SQLite database.");
  }
});

// Create table
db.run(
  `CREATE TABLE IF NOT EXISTS inventory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_name TEXT,
    selling_price REAL,
    date_of_entry TEXT,
    date_of_exit TEXT,
    date_of_expiry TEXT,
    quantity INTEGER,
    cost_price REAL,
    storage_slot TEXT
  )`,
  (err) => {
    if (err) {
      console.error("Error creating table:", err.message);
    }
  }
);

// Fetch inventory
app.get("/api/inventory", (req, res) => {
  db.all("SELECT * FROM inventory", [], (err, rows) => {
    if (err) {
      console.error("Error fetching inventory:", err.message);
      res.status(500).json({ error: err.message });
    } else {
      res.json(rows);
    }
  });
});

// Add inventory
app.post("/api/inventory", (req, res) => {
  const {
    product_name,
    selling_price,
    date_of_entry,
    date_of_exit,
    date_of_expiry,
    quantity,
    cost_price,
    storage_slot,
  } = req.body;

  const query = `INSERT INTO inventory 
    (product_name, selling_price, date_of_entry, date_of_exit, date_of_expiry, quantity, cost_price, storage_slot)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)`;

  const params = [
    product_name,
    selling_price,
    date_of_entry,
    date_of_exit,
    date_of_expiry,
    quantity,
    cost_price,
    storage_slot,
  ];

  db.run(query, params, function (err) {
    if (err) {
      console.error("Database Error:", err.message);
      res.status(500).json({ error: err.message });
    } else {
      console.log("Inserted row ID:", this.lastID);
      res.json({ id: this.lastID });
    }
  });
});

// Start server
app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});