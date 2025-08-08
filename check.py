import socket
import logging

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")


def check_port(ip: str, port: int, timeout=1.0):
    try:
        with socket.create_connection((ip, port), timeout=timeout):
            logging.info(f"[+] Port {port} OPEN")
            return True
    except Exception:
        logging.info(f"[-] Port {port} closed")
        return False


def scan_ports(ip: str, ports: list):
    logging.info(f"Scanning {ip}...")
    for port in ports:
        check_port(ip, port)


if __name__ == "__main__":
    ip = "192.168.4.10"
    ports_to_check = [23, 80, 443, 502, 9760, 8888, 8899, 9999, 6000, 21]
    scan_ports(ip, ports_to_check)
