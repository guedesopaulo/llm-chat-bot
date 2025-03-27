from fpdf import FPDF
import os

products = [
    ("Wireless Mouse", "High precision, ergonomic design", 79.90),
    ("Mechanical Keyboard", "RGB lighting, blue switches", 249.90),
    ("Noise Cancelling Headphones", "Bluetooth, 30h battery", 499.90),
    ("27'' Monitor", "144Hz, 1ms response time", 1099.90),
    ("Webcam Full HD", "1080p, built-in microphone", 199.90),
    ("USB-C Hub", "5 ports, compact design", 89.90),
    ("Gaming Chair", "Reclinable, lumbar support", 899.90),
    ("Portable SSD 1TB", "USB 3.2, up to 1050MB/s", 699.90),
    ("Smartwatch", "Heart rate, sleep tracking", 399.90),
    ("Wireless Charger", "15W fast charging", 129.90)
]

output_folder = "data"
os.makedirs(output_folder, exist_ok=True)
output_path = os.path.join(output_folder, "sale_products.pdf")

pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", "B", 16)
pdf.cell(0, 10, "Products on Sale", ln=True, align='C')

pdf.set_font("Arial", size=12)
pdf.ln(5)

for name, desc, price in products:
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, f"{name} - R$ {price:.2f}", ln=True)
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 8, f"{desc}")
    pdf.ln(2)

pdf.output(output_path)
print(f"✅ PDF saved to: {output_path}")