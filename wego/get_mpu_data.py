import serial
import time
import csv # Import the csv library

# Configure the serial port
port = 'COM19' 
baudrate = 115200

# Initialize a list to store the data
data_points = []
expected_data_points = 512

try:
    # Open the serial connection
    ser = serial.Serial(port, baudrate, timeout=1)
    print(f"Connected to {port} at {baudrate} baud.")
    time.sleep(2)  # Give time for the connection to establish

    print("Waiting for data...")

    while True:
        # Read a line from the serial port
        line = ser.readline().decode('utf-8').strip()

        if line == "Start":
            print("Start of data transmission detected.")
            data_points = []  # Clear the list for a new transmission

        elif line == "End":
            print("End of data transmission detected.")
            print(f"Received {len(data_points)} data points.")
            
            # --- Code to save data to a CSV file ---
            file_name = "mpu6050_data.csv"
            with open(file_name, 'w', newline='') as file:
                writer = csv.writer(file)
                # Optional: Write a header row for all three axes
                writer.writerow(['acceleration_x', 'acceleration_y', 'acceleration_z']) 
                for data_point in data_points:
                    writer.writerow(data_point) # Write the list directly to the CSV
            
            print(f"Data successfully saved to {file_name}.")
            break  # Exit the loop after saving the data

        elif line and line != "":
            try:
                # Convert the line to a float and add it to the list
                values = [float(val.strip()) for val in line.split(',')]
                data_points.append(values)
            except ValueError:
                print(f"Skipping invalid line: {line}")

except serial.SerialException as e:
    print(f"Error: Could not open serial port '{port}'.")
    print(f"Please check that the Arduino is connected and the port is correct. Error details: {e}")

finally:
    if 'ser' in locals() and ser.is_open:
        ser.close()
        print("Serial port closed.")