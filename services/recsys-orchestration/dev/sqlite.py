import sqlite3

# Create SQLite database
conn = sqlite3.connect("gold_layer.db")
cursor = conn.cursor()

# Create a table to represent the Gold layer
cursor.execute("""
CREATE TABLE sales_summary (
    shop_id INTEGER,
    product_id INTEGER,
    category_id INTEGER,
    month TEXT,
    total_sales INTEGER,
    avg_price REAL
)
""")

# Insert sample data
sample_data = [
    (1, 101, 10, "2024-01", 200, 15.5),
    (1, 102, 10, "2024-01", 150, 12.3),
    (2, 101, 10, "2024-01", 300, 14.8),
    (2, 103, 11, "2024-01", 120, 20.0),
    (1, 101, 10, "2024-02", 250, 16.0),
    (1, 102, 10, "2024-02", 180, 13.1),
    (2, 101, 10, "2024-02", 320, 15.2),
    (2, 103, 11, "2024-02", 100, 21.5),
]

cursor.executemany(
    """
INSERT INTO sales_summary (shop_id, product_id, category_id, month, total_sales, avg_price)
VALUES (?, ?, ?, ?, ?, ?)
""",
    sample_data,
)

# Commit and close
conn.commit()
conn.close()

print("SQLite database 'gold_layer.db' created with sample data!")
