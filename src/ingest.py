import paho.mqtt.client as mqtt
import json

# 1. HSL Broker Settings
HOST = "mqtt.hsl.fi"
PORT = 1883
# Subscription for all TRAM vehicle positions
TOPIC = "/hfp/v2/journey/ongoing/vp/tram/#" 

# 2. Callback when we connect to the broker
def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        print("✅ Connected to HSL Broker!")
        client.subscribe(TOPIC)
    else:
        print(f"❌ Failed to connect, return code {rc}")

# 3. Callback when a message (tram data) arrives
def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode('utf-8'))
        # HSL data is nested under 'VP' (Vehicle Position)
        if 'VP' in payload:
            data = payload['VP']
            # Print a summary to the console so you know it's working
            print(f"Tram {data.get('veh')} | Delay: {data.get('dl')}s | Lat: {data.get('lat')}")
            
            # SAVE TO FILE: Append as a new line in a JSONL file
            with open("raw_tram_data.jsonl", "a") as f:
                f.write(json.dumps(data) + "\n")
    except Exception as e:
        print(f"Error parsing JSON: {e}")

# 4. Initialize Client (The 'CallbackAPIVersion' is the critical fix for 2026)
client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
client.on_connect = on_connect
client.on_message = on_message

print("Attempting to connect...")
client.connect(HOST, PORT, 60)

# Start the loop
try:
    client.loop_forever()
except KeyboardInterrupt:
    print("\nStopping data collection...")
    client.disconnect()