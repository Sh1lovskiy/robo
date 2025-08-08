import socket
import json
import time


class BorunteRobotClient:
    def __init__(self, host, port=9760, timeout=5.0):
        self.host = host
        self.port = port
        self.sock = None
        self.timeout = timeout

    def connect(self):
        self.sock = socket.create_connection((self.host, self.port), self.timeout)
        self.sock.settimeout(self.timeout)

    def close(self):
        if self.sock:
            self.sock.close()
            self.sock = None

    def send_json(self, msg: dict):
        data = json.dumps(msg, ensure_ascii=False).encode("utf-8")
        self.sock.sendall(data)

    def recv_json(self):
        buffer = b""
        start_time = time.time()
        while True:
            try:
                chunk = self.sock.recv(4096)
                if not chunk:
                    break
                buffer += chunk
                try:
                    return json.loads(buffer.decode("utf-8"))
                except json.JSONDecodeError:
                    pass
                if time.time() - start_time > self.timeout:
                    break
            except socket.timeout:
                break
        raise TimeoutError("No response or incomplete JSON from robot.")

    def query(self, addrs):
        msg = {
            "dsID": "www.hc-system.com.RemoteMonitor",
            "reqType": "query",
            "packID": "0",
            "queryAddr": addrs if isinstance(addrs, list) else [addrs],
        }
        self.send_json(msg)
        return self.recv_json()

    def clear_alarm(self):
        # Сброс аварий (ALARM)
        msg = {
            "dsID": "www.hc-system.com.RemoteMonitor",
            "reqType": "command",
            "packID": "0",
            "cmdData": ["clearAlarm"],
        }
        self.send_json(msg)
        return self.recv_json()

    def get_alarm(self):
        resp = self.query(["curAlarm"])
        if resp and "queryData" in resp:
            return resp["queryData"][0]
        return None

    def move_tcp_z(self, coords, delta_z, ckstatus="0x3F", speed="10.0"):
        # Linear move (по XYZ), delta_z мм относительно текущей позиции
        move_cmd = {
            "dsID": "www.hc-system.com.HCRemoteCommand",
            "reqType": "AddRCC",
            "emptyList": "1",
            "packID": "0",
            "instructions": [
                {
                    "oneshot": "1",
                    "action": "14",  # linear move
                    "m0": str(coords[0]),  # X
                    "m1": str(coords[1]),  # Y
                    "m2": str(float(coords[2]) + float(delta_z)),  # Z+delta
                    "m3": str(coords[3]),
                    "m4": str(coords[4]),
                    "m5": str(coords[5]),
                    "m6": "0",
                    "m7": "0",
                    "ckStatus": ckstatus,
                    "speed": speed,
                    "delay": "1.0",
                    "coord": "1",
                    "tool": "0",
                    "smooth": "0",
                }
            ],
        }
        self.send_json(move_cmd)
        return self.recv_json()

    def heartbeat(self):
        msg = {"dsID": "www.hc-system.com.RemoteMonitor", "reqType": "heartbreak"}
        self.send_json(msg)


if __name__ == "__main__":
    HOST = "192.168.4.4"
    PORT = 9760

    client = BorunteRobotClient(HOST, PORT)
    try:
        client.connect()
        print("Connected to robot.")

        alarm = client.get_alarm()
        print(f"Current alarm: {alarm}")
        if alarm != "0":
            print("ALARM ACTIVE! Trying to clear...")
            clear_resp = client.clear_alarm()
            print("clearAlarm response:", clear_resp)
            alarm = client.get_alarm()
            if alarm != "0":
                print("ALARM still active! Movement not possible.")
                exit(1)

        mode_resp = client.query(["curMode"])
        mode_val = mode_resp.get("queryData", ["?"])[0]
        print(f"Current robot mode (curMode): {mode_val}")

        addr_list = ["world-0", "world-1", "world-2", "world-3", "world-4", "world-5"]
        tcp = client.query(addr_list)
        coords = tcp.get("queryData", ["0"] * 6)
        print("Start TCP coords:", coords)

        steps = 1
        for i in range(steps):
            print(f"Step {i+1}/{steps}: move TCP on 10 mm down...")
            move_resp = client.move_tcp_z(coords, -50.0, ckstatus="0x3F", speed="30.0")
            print("Response on move:", move_resp)

            if move_resp.get("cmdReply", ["", "ok"])[1] != "ok":
                print("Move error! Breaking loop.")
                break

            time.sleep(2)
            tcp = client.query(addr_list)
            coords = tcp.get("queryData", ["0"] * 6)
            print(f"TCP after step {i+1}:", coords)

        client.heartbeat()

    except Exception as e:
        print("Error:", e)
    finally:
        client.close()
