import serial
import threading
import time
import roslibpy

class LoacellSensor():
    def __init__(self, name, client, serial_port='COM5', baudrate=57600):
        self.name = name
        self.client = client
        self.publisher = roslibpy.Topic(self.client, '/ForceSensor/data', 'std_msgs/String')

        self.ser = serial.Serial(serial_port, baudrate, timeout=1)
        self.ser.flush()
        self.prefix = "Load_cell output val: "
        self.thread = threading.Thread(target=self.read_serial)
        self.thread.daemon = True
        self.thread.start()

        print(f"{self.name} Connected")

    def read_serial(self):
        while self.client.is_connected:
            if self.ser.in_waiting > 0:
                line = self.ser.readline()
                try:
                    line = line.decode('utf-8', errors='ignore').rstrip()
                    if line:
                        print(f'Received: {line}')
                        print(line[len(self.prefix):])
                        reading = float(line[len(self.prefix):])
                        self.publisher.publish(roslibpy.Message({'data': str(reading)}))
                except Exception as e:
                    print(e)

            time.sleep(0.1)  # Adjust the sleep time if needed

    def cleanup(self):
        self.ser.close()
        self.thread.join()
        try:
            if self.ser.is_open:
                self.ser.close()
        except serial.SerialException as e:
            print(f"Error closing serial port: {e}")
        except OSError as e:
            print(f"OS error during serial cleanup: {e}")
        self.publisher.unadvertise()
        self.client.terminate()

if __name__ == "__main__":
    sensor = LoacellSensor()

    try:
        while sensor.client.is_connected:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")
        sensor.cleanup()
