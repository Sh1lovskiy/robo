import xmlrpc.client
import os
import socket
import hashlib
import time
from datetime import datetime
import logging
from functools import wraps
from logging.handlers import RotatingFileHandler
from queue import Queue
import threading
import struct
import sys
import ctypes
from ctypes import *

# from Cython.Compiler.Options import error_on_unknown_names

is_init =False
class ROBOT_AUX_STATE(Structure):
    _pack_ = 1
    _fields_ = [
        ("servoId", c_byte),         # Servo driveIDNumber
        ("servoErrCode", c_int),     # Servo drive fault code
        ("servoState", c_int),       # Servo Drive Status
        ("servoPos", c_double),      # Current position of the servo
        ("servoVel", c_float),       # Current servo speed
        ("servoTorque", c_float),    # Current servo torque
    ]

class EXT_AXIS_STATUS(Structure):
    _pack_ = 1
    _fields_ = [
        ("pos", c_double),        # Extended axis position
        ("vel", c_double),        # Expanded shaft speed
        ("errorCode", c_int),     # Extended axis fault code
        ("ready", c_byte),        # Servo ready
        ("inPos", c_byte),        # Servo in place
        ("alarm", c_byte),        # Servo alarm
        ("flerr", c_byte),        # Follow error
        ("nlimit", c_byte),       # To negative limit
        ("pLimit", c_byte),       # Go to the positive limit
        ("mdbsOffLine", c_byte),  # Drive485Bus drop
        ("mdbsTimeout", c_byte),  # Control card and control box485Communication timeout
        ("homingStatus", c_byte), # Expanded axis back to zero state
    ]

class WELDING_BREAKOFF_STATE(Structure):
    _pack_ = 1
    _fields_ = [
        ("breakOffState", ctypes.c_uint8),        # Welding interrupted state
        ("weldArcState", ctypes.c_uint8),        # Welding arc interruption status
    ]

class RobotStatePkg(Structure):
    _pack_ = 1
    _fields_ = [
        ("frame_head", c_uint16),      # Frame header 0x5A5A
        ("frame_cnt", c_byte),         # Frame count
        ("data_len", c_uint16),        # Data length
        ("program_state", c_byte),     # Program running status，1-stop；2-run；3-pause
        ("robot_state", c_byte),       # Robot motion state，1-stop；2-run；3-pause；4-drag
        ("main_code", c_int),          # Main fault code
        ("sub_code", c_int),           # Sub-fault code
        ("robot_mode", c_byte),        # Robot mode，0-Automatic mode；1-Manual mode
        ("jt_cur_pos", c_double * 6),  # Current joint position of the robot，Assume there is6A joint
        ("tl_cur_pos", ctypes.c_double * 6),  # Tool Current Pose
        ("flange_cur_pos", ctypes.c_double * 6),  # Current position of the end flange
        ("actual_qd", ctypes.c_double * 6),  # The current joint speed of the robot
        ("actual_qdd", ctypes.c_double * 6),  # The robot's current joint acceleration
        ("target_TCP_CmpSpeed", ctypes.c_double * 2),  # robotTCPSynthesis command speed
        ("target_TCP_Speed", ctypes.c_double * 6),  # robotTCPCommand speed
        ("actual_TCP_CmpSpeed", ctypes.c_double * 2),  # robotTCPActual synthesis speed
        ("actual_TCP_Speed", ctypes.c_double * 6),  # robotTCPActual speed
        ("jt_cur_tor", ctypes.c_double * 6),  # Current torque
        ("tool", ctypes.c_int),  # Tool number
        ("user", ctypes.c_int),  # Workpiece number
        ("cl_dgt_output_h", ctypes.c_byte),  # Digital output15-8
        ("cl_dgt_output_l", ctypes.c_byte),  # Digital output7-0
        ("tl_dgt_output_l", ctypes.c_byte),  # Tool digital output7-0(onlybit0-bit1efficient)
        ("cl_dgt_input_h", ctypes.c_byte),  # Digital input15-8
        ("cl_dgt_input_l", ctypes.c_byte),  # Digital input7-0
        ("tl_dgt_input_l", ctypes.c_byte),  # Tool digital input7-0(onlybit0-bit1efficient)
        ("cl_analog_input", ctypes.c_uint16 * 2),  # Control box analog input
        ("tl_anglog_input", ctypes.c_uint16),  # Tool analog input
        ("ft_sensor_raw_data", ctypes.c_double * 6),  # force/Torque sensor raw data
        ("ft_sensor_data", ctypes.c_double * 6),  # force/Torque sensor data
        ("ft_sensor_active", ctypes.c_byte),  # force/Torque sensor activation status， 0-Reset，1-activation
        ("EmergencyStop", ctypes.c_byte),  # Emergency stop sign
        ("motion_done", ctypes.c_int),  # In-place signal
        ("gripper_motiondone", ctypes.c_byte),  # Claw movement completion signal
        ("mc_queue_len", ctypes.c_int),  # Sports queue length
        ("collisionState", ctypes.c_byte),  # Collision detection，1-collision；0-No collision
        ("trajectory_pnum", ctypes.c_int),  # Track point number
        ("safety_stop0_state", ctypes.c_byte),  # Safe stop signalSI0
        ("safety_stop1_state", ctypes.c_byte),  # Safe stop signalSI1
        ("gripper_fault_id", ctypes.c_byte),  # Error claw number
        ("gripper_fault", ctypes.c_uint16),  # Claw failure
        ("gripper_active", ctypes.c_uint16),  # Claw activation status
        ("gripper_position", ctypes.c_byte),  # Claw position
        ("gripper_speed", ctypes.c_byte),  # Claw speed
        ("gripper_current", ctypes.c_byte),  # Claw current
        ("gripper_tmp", ctypes.c_int),  # Claw temperature
        ("gripper_voltage", ctypes.c_int),  # Claw voltage
        ("auxState", ROBOT_AUX_STATE),  # 485Extended axis status
        ("extAxisStatus", EXT_AXIS_STATUS*4),  # UDPExtended axis status
        ("extDIState", c_uint16*8),  # ExtendedDIenter
        ("extDOState", c_uint16*8),  # ExtendedDOOutput
        ("extAIState", c_uint16*4),  # ExtendedAIenter
        ("extAOState", c_uint16*4),  # ExtendedAOOutput
        ("rbtEnableState", ctypes.c_int),  # Robot enable status
        ("jointDriverTorque", ctypes.c_double * 6),  # Current torque of joint drive
        ("jointDriverTemperature", ctypes.c_double * 6),  # Current temperature of joint drive
        ("year", ctypes.c_uint16),  # Year
        ("mouth", ctypes.c_uint8),  # moon
        ("day", ctypes.c_uint8),  # day
        ("hour", ctypes.c_uint8),  # Hour
        ("minute", ctypes.c_uint8),  # point
        ("second", ctypes.c_uint8),  # Second
        ("millisecond", ctypes.c_uint16),  # millisecond
        ("softwareUpgradeState", ctypes.c_int),  # Robot software upgrade status
        ("endLuaErrCode", ctypes.c_uint16),  # EndLUARunning status
        ("cl_analog_output", ctypes.c_uint16 * 2),  # Control box analog output
        ("tl_analog_output", ctypes.c_uint16),  # Tool analog output
        ("gripperRotNum", ctypes.c_float),  # Current rotation number of rotations of rotating jaws
        ("gripperRotSpeed", ctypes.c_uint8),  # Current rotation speed percentage of rotating jaws
        ("gripperRotTorque", ctypes.c_uint8),  # Current rotation torque percentage of rotating jaw
        ("weldingBreakOffState", WELDING_BREAKOFF_STATE), # Welding interrupted state
        ("check_sum", c_ushort)]  # Checksum


class BufferedFileHandler(RotatingFileHandler):
    def __init__(self, filename, mode='a', maxBytes=0, backupCount=0, encoding=None, delay=False):
        super().__init__(filename, mode, maxBytes, backupCount, encoding, delay)
        self.buffer = []

    def emit(self, record):
        # log_entry = self.format(record)  # Format logging
        # print(log_entry)  # Print log entries
        if RPC.log_output_model == 2:
            RPC.queue.put(record)
        else:
            self.buffer.append(record)
            if len(self.buffer) >= 50:
                for r in self.buffer:
                    super().emit(r)
                self.buffer = []


class LogWriterThread(threading.Thread):
    def __init__(self, queue, log_handler):
        super().__init__()
        self.queue = queue
        self.log_handler = log_handler
        self.daemon = True

    def run(self):
        while True:
            record = self.queue.get()
            if record is None:
                break
            log_entry = self.log_handler.format(record)
            self.log_handler.stream.write(log_entry + self.log_handler.terminator)
            self.log_handler.flush()


def calculate_file_md5(file_path):
    if not os.path.exists(file_path):
        raise ValueError(f"{file_path} Does not exist")
    md5 = hashlib.md5()
    with open(file_path, 'rb') as file:
        while chunk := file.read(8192):  # Read in 8KB chunks
            md5.update(chunk)
    return md5.hexdigest()


def xmlrpc_timeout(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if RPC.is_conect == False:
            return -4
        else:
            result = func(self, *args, **kwargs)
            return result

    return wrapper


class RobotError:
    ERR_SUCCESS = 0
    ERR_POINTTABLE_NOTFOUND = -7  # The uploaded file does not exist
    ERR_SAVE_FILE_PATH_NOT_FOUND = -6  # The save file path does not exist
    ERR_NOT_FOUND_LUA_FILE = -5  # luaThe file does not exist
    ERR_RPC_ERROR = -4
    ERR_SOCKET_COM_FAILED = -2
    ERR_OTHER = -1


class RPC():
    ip_address = "192.168.58.2"

    logger = None
    log_output_model = -1
    queue = Queue(maxsize=10000 * 1024)
    logging_thread = None
    is_conect = True
    ROBOT_REALTIME_PORT = 20004
    # BUFFER_SIZE = 1024 * 8
    BUFFER_SIZE = 1024 * 1024
    thread=  threading.Thread()
    SDK_state=True

    sock_cli_state_state = False
    closeRPC_state = False


    def __init__(self, ip="192.168.58.2"):
        self.ip_address = ip
        link = 'http://' + self.ip_address + ":20003"
        self.robot = xmlrpc.client.ServerProxy(link)#xmlrpcConnecting the robot20003port，Used to send robot instruction data frames

        self.sock_cli_state = None
        self.robot_realstate_exit = False
        self.robot_state_pkg = RobotStatePkg#Robot status data

        self.stop_event = threading.Event()  # Stop Event
        self.connect_to_robot()
        thread= threading.Thread(target=self.robot_state_routine_thread)#Create a thread to loop to receive robot status data
        thread.daemon = True
        thread.start()
        time.sleep(1)
        print(self.robot)


        try:
            # Call XML-RPC method
            socket.setdefaulttimeout(1)
            self.robot.GetControllerIP()
        except socket.timeout:
            print("XML-RPC connection timed out.")
            RPC.is_conect = False

        except socket.error as e:
            print("It may be a network failure，Please check the network connection。")
            RPC.is_conect = False
        except Exception as e:
            print("An error occurred during XML-RPC call:", e)
            RPC.is_conect = False
        finally:
            # Restore default timeout
            self.robot = None
            socket.setdefaulttimeout(None)
            self.robot = xmlrpc.client.ServerProxy(link)
    def connect_to_robot(self):
        """Connect to the robot's real-time port"""
        print("SDKConnecting the robot")
        self.sock_cli_state = socket.socket(socket.AF_INET, socket.SOCK_STREAM)#Socket connection robot20004port，Used to update robot status data in real time
        try:
            self.sock_cli_state.connect((self.ip_address, self.ROBOT_REALTIME_PORT))
            self.sock_cli_state_state = True
        except Exception as ex:
            self.sock_cli_state_state = False
            print("SDKFailed to connect to the real-time port of the robot", ex)
            return False
        return True

    def reconnect(self):
        """Automatic reconnection"""
        print("Automatic reconnect mechanism")
        for i in range(1,6):
            time.sleep(2)
            if(self.connect_to_robot()):
                self.sock_cli_state_state = True
                return

    def robot_state_routine_thread(self):
        """Threading routines that handle robot state packets"""

        while not self.closeRPC_state:
            recvbuf = bytearray(self.BUFFER_SIZE)
            tmp_recvbuf = bytearray(self.BUFFER_SIZE)
            state_pkg = bytearray(self.BUFFER_SIZE)
            find_head_flag = False
            index = 0
            length = 0
            tmp_len = 0
            # if not self.sock_cli_state_state:
            #     if not self.connect_to_robot():
            #         return

            try:
                # while not self.robot_realstate_exit:
                while not self.robot_realstate_exit and not self.stop_event.is_set():
                    recvbyte = self.sock_cli_state.recv_into(recvbuf)
                    # timestamp_ms = int(datetime.now().timestamp() * 1000)
                    # print("Current timestamp（Millisecond level）:", timestamp_ms)
                    if recvbyte <= 0:
                        self.sock_cli_state.close()
                        print("Receive robot status bytes -1")
                        return
                    else:
                        if tmp_len > 0:
                            if tmp_len + recvbyte <= self.BUFFER_SIZE:
                                recvbuf = tmp_recvbuf[:tmp_len] + recvbuf[:recvbyte]
                                recvbyte += tmp_len
                                tmp_len = 0
                            else:
                                tmp_len = 0

                        for i in range(recvbyte):
                            if format(recvbuf[i], '02X') == "5A" and not find_head_flag:
                                if i + 4 < recvbyte:
                                    if format(recvbuf[i+1], '02X') == "5A":
                                        find_head_flag = True
                                        state_pkg[0] = recvbuf[i]
                                        index += 1
                                        length = length | recvbuf[i + 4]
                                        length = length << 8
                                        length = length | recvbuf[i + 3]
                                    else:
                                        continue
                                else:
                                    tmp_recvbuf[:recvbyte - i] = recvbuf[i:recvbyte]
                                    tmp_len = recvbyte - i
                                    break
                            elif find_head_flag and index < length + 5:
                                state_pkg[index] = recvbuf[i]
                                index += 1
                            elif find_head_flag and index >= length + 5:
                                if i + 1 < recvbyte:
                                    checksum = sum(state_pkg[:index])
                                    checkdata = 0
                                    checkdata = checkdata | recvbuf[i + 1]
                                    checkdata = checkdata << 8
                                    checkdata = checkdata | recvbuf[i]

                                    if checksum == checkdata:
                                        self.robot_state_pkg = RobotStatePkg.from_buffer_copy(recvbuf)
                                        find_head_flag = False
                                        index = 0
                                        length = 0
                                        i += 1
                                    else:
                                        # print(checksum,":",checkdata,"===========================")
                                        self.robot_state_pkg.jt_cur_pos[0] = 0
                                        self.robot_state_pkg.jt_cur_pos[1] = 0
                                        self.robot_state_pkg.jt_cur_pos[2] = 0
                                        find_head_flag = False
                                        index = 0
                                        length = 0
                                        i += 1
                                else:
                                    print("4.2")
                                    tmp_recvbuf[:recvbyte - i] = recvbuf[i:recvbyte]
                                    tmp_len = recvbyte - i
                                    break
                            else:
                                continue
            except Exception as ex:
                if not self.closeRPC_state:
                    self.sock_cli_state.close()
                    self.sock_cli_state_state = False
                    self.SDK_state=False
                    # self.reconnect()
                    print("SDKFailed to read real-time data of the robot", ex)

    def setup_logging(self, output_model=1, file_path="", file_num=5):
        """Used to process logs"""
        self.logger = logging.getLogger("RPCLogger")
        log_level = logging.DEBUG
        log_handler = None

        if not file_path:
            current_datetime = datetime.now()
            formatted_date = current_datetime.strftime("%Y%m%d")
            file_name = "fairino_" + formatted_date + ".log"
            file_path = os.path.join(os.getcwd(), file_name)  # Use the current working directory，If no path is provided
        else:
            file_path = os.path.abspath(file_path)  # Get the absolute path

        # Check if the directory exists
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            # print(f"Error: The directory '{directory}' does not exist. Logging setup aborted.")
            return -1  # If the directory does not exist，Returns the error code

        if output_model == 0:
            RPC.log_output_model = 0
            log_handler = RotatingFileHandler(file_path, maxBytes=50 * 1024, backupCount=file_num)
        elif output_model == 1:
            RPC.log_output_model = 1
            log_handler = BufferedFileHandler(file_path, mode='a', maxBytes=50 * 1024, backupCount=file_num)
        elif output_model == 2:
            RPC.log_output_model = 2
            log_handler = BufferedFileHandler(file_path, mode='a', maxBytes=50 * 1024, backupCount=file_num)
            self.start_logging_thread(log_handler)

        formatter = logging.Formatter('[%(levelname)s] [%(asctime)s pid:%(process)d]  %(message)s')
        if log_handler:
            log_handler.setFormatter(formatter)
            self.logger.addHandler(log_handler)
        else:
            print("Error: Log handler not created. Logging setup aborted.")

        return 0  # If logging is set successfully，Return the success code

    def start_logging_thread(self, log_handler):
        """Create threads for log storage"""
        logging_thread = LogWriterThread(RPC.queue, log_handler)
        RPC.logging_thread = logging_thread  # Store references to log threads
        logging_thread.start()

    def join_logging_thread(self):
        """Notify the log thread to stop"""
        if RPC.logging_thread is not None:
            RPC.queue.put(None)  # Notify the log thread to stop
            RPC.logging_thread.join()  # Wait for the log thread to complete

    def __del__(self):
        """Garbage collector，Similar to destruction"""
        self.join_logging_thread()#Stop log thread

    def set_log_level(self, lvl):
        """Set logging level"""
        levels = {1: logging.ERROR, 2: logging.WARNING, 3: logging.INFO, 4: logging.DEBUG}
        log_level = levels.get(lvl, logging.DEBUG)
        self.logger.setLevel(log_level)
        return log_level

    def log_call(func):
        """Logging function calls"""
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            args_str = ', '.join(map(repr, args))
            kwargs_str = ', '.join([f"{key}={value}" for key, value in kwargs.items()])
            if (kwargs_str) == "":
                call_message = f"Calling {func.__name__}" + f"({args_str}" + ")."
            else:
                call_message = f"Calling {func.__name__}" + f"({args_str}" + "," + f"{kwargs_str})."

            self.log_info(call_message)
            result = func(self, *args, **kwargs)
            if isinstance(result, (list, tuple)) and len(result) > 0:
                if result[0] == 0:
                    self.log_debug(f"{func.__name__} returned: {result}.")
                else:
                    self.log_error(f"{func.__name__} Error occurred. returned: {result}")
            else:
                if result == 0:
                    self.log_debug(f"{func.__name__} returned: {result}.")
                else:
                    self.log_error(f"{func.__name__} Error occurred. returned: {result}")

            return result

        return wrapper

    def log_debug(self, message):
        """Used for recordingdebugLevel log"""
        if self.logger:
            self.logger.debug(message)

    def log_info(self, message):
        """Used for recordinginfoLevel log"""
        if self.logger:
            self.logger.info(message)

    def log_warning(self, message):
        """Used for recordingwarningLevel log"""
        if self.logger:
            self.logger.warning(message)

    def log_error(self, message):
        """Used for recordingerrrorLevel log"""
        if self.logger:
            self.logger.error(message)

    def send_message(self, message):
        """createtcpConnect to send a message"""
        # Create aTCP/IPSockets
        sock1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        port = 8080  # Fixed port number is8080
        try:
            # Connect to the server
            sock1.connect((self.ip_address, 8080))
            # Send data
            sock1.sendall(message.encode('utf-8'))

            response = sock1.recv(1024).decode('utf-8')

            value =response.split('III')
            if len(value) ==6:
                if value[4] == "1":
                    return 0
                else:
                    print("error happended",value[4])
                    return -1
            else:
                return -1
        except Exception as e:
            print(f'An error occurred: {e}')

        finally:
            sock1.close()

    """2024.12.23"""
    """   
       @brief Security code acquisition
       @return Error code success- 0, fail-Error code
    """

    def GetSafetyCode(self):
        if (self.robot_state_pkg.safety_stop0_state == 1) or (self.robot_state_pkg.safety_stop1_state == 1):
            return 99
        return 0
    """2024.12.23"""
    """   
    ***************************************************************************Robot basics********************************************************************************************
    """

    """   
    @brief  Query SDK Version number
    @param  [in] NULL
    @return Error code success-0  fail-Error code
    @return Return value（The call returns successfully） version SDKVersion number
    """

    @log_call
    @xmlrpc_timeout
    def GetSDKVersion(self):
        error = 0
        sdk = ["SDK:V2.0.8", "Robot:V3.7.8"]
        return error, sdk

    """   
    @brief  Query Controller IP
    @param  [in] NULL
    @return Error code success-0,  fail-Error code
    @return Return value（The call returns successfully） ip  ControllerIP
    """

    @log_call
    @xmlrpc_timeout
    def GetControllerIP(self):
        _error = self.robot.GetControllerIP()
        error = _error[0]
        if _error[0] == 0:
            return error, _error[1]
        else:
            return error

    """   
    @brief  Control the robot's manual automatic mode switching
    @param  [in] Required parameters state：0-Automatic mode 1-Manual mode
    @return Error code success-0  fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def Mode(self, state):
        state = int(state)
        error = self.robot.Mode(state)
        return error

    """   
    @brief  Control the robot to enter or exit drag teaching mode
    @param  [in] Required parameters state：0-Exit drag teaching mode, 1-Enter drag teaching mode
    @return Error code success-0  fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def DragTeachSwitch(self, state):
        state = int(state)  # Cast tointtype
        error = self.robot.DragTeachSwitch(state)
        return error

    """   
    @brief  Check whether the robot is in drag teaching mode
    @param  [in] NULL
    @return Error code success-0，fail-Error code
    @return Return value（The call returns successfully） state 0-Non-drag teaching mode，1-Drag teaching mode
    """

    @log_call
    @xmlrpc_timeout
    def IsInDragTeach(self):
        _error = self.robot.IsInDragTeach()
        error = _error[0]
        if _error[0] == 0:
            return error, _error[1]
        else:
            return error

    """   
    @brief  Control the robot to enable or enable
    @param  [in] Required parameters state：0-Enable, 1-Enable on
    @return Error code success-0  fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def RobotEnable(self, state):
        state = int(state)  # Cast tointtype
        error = self.robot.RobotEnable(state)
        return error

    """   
    ***************************************************************************Robot movement********************************************************************************************
    """

    """   
    @brief  jogJog
    @param  [in] Required parameters ref：0-The node movement,2-Base coordinate system jog,4-Tool coordinate system jog,8-Jogging of workpiece coordinate system
    @param  [in] Required parameters nb：1-joint1(orxaxis)，2-joint2(oryaxis)，3-joint3(orzaxis)，4-joint4(Or aroundxAxis rotation)，5-joint5(Or aroundyAxis rotation)，6-joint6(Or aroundzAxis rotation)
    @param  [in] Required parameters dir：0-Negative direction，1-Positive direction
    @param  [in] Must recognize parameters max_dis：Maximum angle for single jog/distance，unit ° or mm
    @param  [in] Default parameters vel：Speed ​​percentage，[0~100] default20
    @param  [in] Default parameters acc：Acceleration percentage，[0~100] default100
    @return Error code success-0  fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def StartJOG(self, ref, nb, dir, max_dis, vel=20.0, acc=100.0):
        if self.GetSafetyCode() != 0:
            return self.GetSafetyCode()
        ref = int(ref)  # Cast tointtype
        nb = int(nb)  # Cast tointtype
        dir = int(dir)  # Cast tointtype
        max_dis = float(max_dis)  # Cast tofloattype
        vel = float(vel)  # Cast tofloattype
        acc = float(acc)  # Cast tofloattype
        error = self.robot.StartJOG(ref, nb, dir, vel, acc, max_dis)
        return error

    """   
    @brief  jog Jog slow down stop
    @param  [in] Required parameters：1-The node stops,3-The base coordinate system jog stops,5-Tool coordinate system jog stops,9-Jogging stops workpiece coordinate system
    @return Error code success-0  fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def StopJOG(self, ref):
        ref = int(ref)  # Cast tointtype
        error = self.robot.StopJOG(ref)
        return error

    """   
    @brief  jog Stop immediately by jogging
    @param  [in] NULL
    @return Error code success-0  fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def ImmStopJOG(self):
        error = self.robot.ImmStopJOG()
        return error

    """   
    @brief  Joint space movement
    @param  [in] Required parameters joint_pos: Target joint position，unit [°]
    @param  [in] Required parameters tool: Tool number，[0~14]
    @param  [in] Required parameters user: Workpiece number，[0~14]
    @param  [in] Default parameters desc_pos: Target Cartesian pose，unit [mm][°] The default initial value is[0.0,0.0,0.0,0.0,0.0,0.0]，The default value calls positive kinematics solution to return value
    @param  [in] Default parameters vel：Speed ​​percentage，[0~100] default20.0
    @param  [in] Default parameters acc：Acceleration percentage，[0~100] Not open yet,default0.0 
    @param  [in] Default parameters ovl: Speed ​​scaling factor，[0~100] default100.0
    @param  [in] Default parameters exaxis_pos: External axis 1 Location ~ External axis 4 Location default[0.0,0.0,0.0,0.0]
    @param  [in] Default parameters blendT:[-1.0]-Exercise in place (block)，[0~500.0]-Smoothing time (Non-blocking)，unit [ms] default-1.0
    @param  [in] Default parameters offset_flag:[0]-No offset，[1]-Workpieces/Offset under base coordinate system，[2]-Offset under tool coordinate system default 0
    @param  [in] Default parameters offset_pos: Pose offset，unit [mm][°] default[0.0,0.0,0.0,0.0,0.0,0.0]
    @return Error code success-0  fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def MoveJ(self, joint_pos, tool, user, desc_pos=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], vel=20.0, acc=0.0, ovl=100.0,
              exaxis_pos=[0.0, 0.0, 0.0, 0.0], blendT=-1.0, offset_flag=0, offset_pos=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]):
        if self.GetSafetyCode() != 0:
            return self.GetSafetyCode()
        joint_pos = list(map(float, joint_pos))
        tool = int(tool)
        user = int(user)
        desc_pos = list(map(float, desc_pos))
        vel = float(vel)
        acc = float(acc)
        ovl = float(ovl)
        exaxis_pos = list(map(float, exaxis_pos))
        blendT = float(blendT)
        offset_flag = int(offset_flag)
        offset_pos = list(map(float, offset_pos))
        if (desc_pos[0] == 0.0) and (desc_pos[1] == 0.0) and (desc_pos[2] == 0.0) and (desc_pos[3] == 0.0) and (
                desc_pos[4] == 0.0) and (desc_pos[5] == 0.0):  # If no parameters are entered, call positive kinematics to solve
            ret = self.robot.GetForwardKin(joint_pos)  # Solve the kinematics
            if ret[0] == 0:
                desc_pos = [ret[1], ret[2], ret[3], ret[4], ret[5], ret[6]]
            else:
                error = ret[0]
                return error
        error = self.robot.MoveJ(joint_pos, desc_pos, tool, user, vel, acc, ovl, exaxis_pos, blendT, offset_flag,
                                 offset_pos)
        return error

    """   
    @brief  Cartesian space linear motion
    @param  [in] Required parameters desc_pos: Target Cartesian pose，unit [mm][°]
    @param  [in] Required parameters tool: Tool number，[0~14]
    @param  [in] Required parameters user: Workpiece number，[0~14]
    @param  [in] Default parameters joint_pos: Target joint position，unit [°] The default initial value is[0.0,0.0,0.0,0.0,0.0,0.0]，The default value calls inverse kinematics solution to return value
    @param  [in] Default parameters vel：Speed ​​percentage，[0~100] default20.0
    @param  [in] Default parameters acc：Acceleration percentage，[0~100] Not open yet default0.0
    @param  [in] Default parameters ovl: Speed ​​scaling factor，[0~100] default100.0
    @param  [in] Default parameters blendR:[-1.0]-Exercise in place (block)，[0~1000]-Smooth radius (Non-blocking)，unit [mm] default-1.0
    @param  [in] Default parameters exaxis_pos: External axis 1 Location ~ External axis 4 Location default[0.0,0.0,0.0,0.0]
    @param  [in] Default parameters search:[0]-No welding wire position search，[1]-Wire positioning
    @param  [in] Default parameters offset_flag:[0]-No offset，[1]-Workpieces/Offset under base coordinate system，[2]-Offset under tool coordinate system default 0
    @param  [in] Default parameters offset_pos: Pose offset，unit [mm][°] default[0.0,0.0,0.0,0.0,0.0,0.0]
    @param  [in] Default parameters overSpeedStrategy  Speeding processing strategy，0-Policy Close；1-standard；2-Speeding error stop；3-Adaptive speed reduction，Default is0
    @param  [in] Default parameters speedPercent  Permitted speed down threshold percentage[0-100]，default10%
    @return Error code success-0  fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def MoveL(self, desc_pos, tool, user, joint_pos=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], vel=20.0, acc=0.0, ovl=100.0,
              blendR=-1.0,exaxis_pos=[0.0, 0.0, 0.0, 0.0], search=0, offset_flag=0,
              offset_pos=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],overSpeedStrategy=0,speedPercent=10):
        if self.GetSafetyCode() != 0:
            return self.GetSafetyCode()
        desc_pos = list(map(float, desc_pos))
        tool = int(tool)
        user = int(user)
        joint_pos = list(map(float, joint_pos))
        vel = float(vel)
        acc = float(acc)
        ovl = float(ovl)
        blendR = float(blendR)
        exaxis_pos = list(map(float, exaxis_pos))
        search = int(search)
        offset_flag = int(offset_flag)
        offset_pos = list(map(float, offset_pos))
        overSpeedStrategy = int(overSpeedStrategy)
        speedPercent = int(speedPercent)
        if (overSpeedStrategy > 0):
            error = self.robot.JointOverSpeedProtectStart(overSpeedStrategy, speedPercent)
            if error!=0:
                return error
        if ((joint_pos[0] == 0.0) and (joint_pos[1] == 0.0) and (joint_pos[2] == 0.0) and (joint_pos[3] == 0.0)
                and (joint_pos[4] == 0.0) and (joint_pos[5] == 0.0)):  # If no parameters are entered, inverse kinematics is called to solve the problem.
            ret = self.robot.GetInverseKin(0, desc_pos, -1)  # Resolving inverse kinematics
            if ret[0] == 0:
                joint_pos = [ret[1], ret[2], ret[3], ret[4], ret[5], ret[6]]
            else:
                error1 = ret[0]
                return error1
        error1 = self.robot.MoveL(joint_pos, desc_pos, tool, user, vel, acc, ovl, blendR, exaxis_pos, search,
                                 offset_flag, offset_pos)
        if (overSpeedStrategy > 0):
            error = self.robot.JointOverSpeedProtectEnd()
            if error!=0:
                return error

        return error1

    """   
    @brief  Cartesian space arc motion
    @param  [in] Required parameters desc_pos_p: Cartesian pose，unit [mm][°]
    @param  [in] Required parameters tool_p: Pathpoint tool number，[0~14]
    @param  [in] Required parameters user_p: Path point workpiece number，[0~14]
    @param  [in] Required parameters desc_pos_t: Cartesian pose of the target point，unit [mm][°]
    @param  [in] Required parameters tool_t: Tool number，[0~14]
    @param  [in] Required parameters user_t: Workpiece number，[0~14]
    @param  [in] Default parameters joint_pos_p: Path point joint position，unit [°] The default initial value is[0.0,0.0,0.0,0.0,0.0,0.0]，The default value calls inverse kinematics solution to return value
    @param  [in] Default parameters joint_pos_t: Target joint position，unit [°] The default initial value is[0.0,0.0,0.0,0.0,0.0,0.0]，The default value calls inverse kinematics solution to return value
    @param  [in] Default parameters vel_p: Path point velocity percentage，[0~100] default20.0
    @param  [in] Default parameters acc_p: Path point acceleration percentage，[0~100] Not open yet,default0.0
    @param  [in] Default parameters exaxis_pos_p: Path point external axis 1 Location ~ External axis 4 Location default[0.0,0.0,0.0,0.0]
    @param  [in] Default parameters offset_flag_p: Is the path point offset?[0]-No offset，[1]-Workpieces/Offset under base coordinate system，[2]-Offset under tool coordinate system default 0
    @param  [in] Default parameters offset_pos_p: Path Point Position Offset，unit [mm][°] default[0.0,0.0,0.0,0.0,0.0,0.0]
    @param  [in] Default parameters vel_t: Target point speed percentage，[0~100] default20.0
    @param  [in] Default parameters acc_t: Target point acceleration percentage，[0~100] Not open yet default0.0
    @param  [in] Default parameters exaxis_pos_t: Target point external axis 1 Location ~ External axis 4 Location default[0.0,0.0,0.0,0.0]
    @param  [in] Default parameters offset_flag_t: Is the target point offset?[0]-No offset，[1]-Workpieces/Offset under base coordinate system，[2]-Offset under tool coordinate system default 0
    @param  [in] Default parameters offset_pos_t: Target Point Position Offset，unit [mm][°] default[0.0,0.0,0.0,0.0,0.0,0.0]
    @param  [in] Default parameters ovl: Speed ​​scaling factor，[0~100] default100.0
    @param  [in] Default parameters blendR:[-1.0]-Exercise in place (block)，[0~1000]-Smooth radius (Non-blocking)，unit [mm] default-1.0
    @return Error code success-0  fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def MoveC(self, desc_pos_p, tool_p, user_p, desc_pos_t, tool_t, user_t, joint_pos_p=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              joint_pos_t=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              vel_p=20.0, acc_p=100.0, exaxis_pos_p=[0.0, 0.0, 0.0, 0.0], offset_flag_p=0,
              offset_pos_p=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              vel_t=20.0, acc_t=100.0, exaxis_pos_t=[0.0, 0.0, 0.0, 0.0], offset_flag_t=0,
              offset_pos_t=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              ovl=100.0, blendR=-1.0):
        if self.GetSafetyCode() != 0:
            return self.GetSafetyCode()
        desc_pos_p = list(map(float, desc_pos_p))
        tool_p = float(int(tool_p))
        user_p = float(int(user_p))
        joint_pos_p = list(map(float, joint_pos_p))
        vel_p = float(vel_p)
        acc_p = float(acc_p)
        exaxis_pos_p = list(map(float, exaxis_pos_p))
        offset_flag_p = int(offset_flag_p)
        offset_pos_p = list(map(float, offset_pos_p))

        desc_pos_t = list(map(float, desc_pos_t))
        tool_t = float(int(tool_t))
        user_t = float(int(user_t))
        joint_pos_t = list(map(float, joint_pos_t))
        vel_t = float(vel_t)
        acc_t = float(acc_t)
        exaxis_pos_t = list(map(float, exaxis_pos_t))
        offset_flag_t = int(offset_flag_t)
        offset_pos_t = list(map(float, offset_pos_t))

        ovl = float(ovl)
        blendR = float(blendR)

        if ((joint_pos_p[0] == 0.0) and (joint_pos_p[1] == 0.0) and (joint_pos_p[2] == 0.0) and (joint_pos_p[3] == 0.0)
                and (joint_pos_p[4] == 0.0) and (joint_pos_p[5] == 0.0)):  # If no parameters are entered, inverse kinematics is called to solve the problem.
            retp = self.robot.GetInverseKin(0, desc_pos_p, -1)  # Resolving inverse kinematics
            if retp[0] == 0:
                joint_pos_p = [retp[1], retp[2], retp[3], retp[4], retp[5], retp[6]]
            else:
                error = retp[0]
                return error

        if ((joint_pos_t[0] == 0.0) and (joint_pos_t[1] == 0.0) and (joint_pos_t[2] == 0.0) and (joint_pos_t[3] == 0.0)
                and (joint_pos_t[4] == 0.0) and (joint_pos_t[5] == 0.0)):  # If no parameters are entered, inverse kinematics is called to solve the problem.
            rett = self.robot.GetInverseKin(0, desc_pos_t, -1)  # Resolving inverse kinematics
            if rett[0] == 0:
                joint_pos_t = [rett[1], rett[2], rett[3], rett[4], rett[5], rett[6]]
            else:
                error = rett[0]
                return error
        error = self.robot.MoveC(joint_pos_p, desc_pos_p, [tool_p, user_p, vel_p, acc_p], exaxis_pos_p, offset_flag_p,
                                 offset_pos_p, joint_pos_t, desc_pos_t, [tool_t, user_t, vel_t, acc_t], exaxis_pos_t,
                                 offset_flag_t, offset_pos_t, ovl, blendR)
        return error

    """   
    @brief  Cartesian space Circular Movement
    @param  [in] Required parameters desc_pos_p: Cartesian pose，unit [mm][°]
    @param  [in] Required parameters tool_p: Pathpoint tool number，[0~14]
    @param  [in] Required parameters user_p: Path point workpiece number，[0~14]
    @param  [in] Required parameters desc_pos_t: Cartesian pose of the target point，unit [mm][°]
    @param  [in] Required parameters tool_t: Tool number，[0~14]
    @param  [in] Required parameters user_t: Workpiece number，[0~14]
    @param  [in] Default parameters joint_pos_p: Path point joint position，unit [°] The default initial value is[0.0,0.0,0.0,0.0,0.0,0.0]，The default value calls inverse kinematics solution to return value
    @param  [in] Default parameters joint_pos_t: Target joint position，unit [°] The default initial value is[0.0,0.0,0.0,0.0,0.0,0.0]，The default value calls inverse kinematics solution to return value
    @param  [in] Default parameters vel_p: Path point velocity percentage，[0~100] default20.0
    @param  [in] Default parameters acc_p: Path point acceleration percentage，[0~100] Not open yet default0.0
    @param  [in] Default parameters exaxis_pos_p: Path point external axis 1 Location ~ External axis 4 Location default[0.0,0.0,0.0,0.0]
    @param  [in] Default parameters vel_t: Target point speed percentage，[0~100] default20.0
    @param  [in] Default parameters acc_t: Target point acceleration percentage，[0~100] Not open yet default0.0
    @param  [in] Default parameters exaxis_pos_t: Target point external axis 1 Location ~ External axis 4 Location default[0.0,0.0,0.0,0.0]
    @param  [in] Default parameters ovl: Speed ​​scaling factor，[0~100] default100.0
    @param  [in] Default parameters offset_flag: Whether offset[0]-No offset，[1]-Workpieces/Offset under base coordinate system，[2]-Offset under tool coordinate system default 0
    @param  [in] Default parameters offset_pos: Pose offset，unit [mm][°] default[0.0,0.0,0.0,0.0,0.0,0.0]
    @return Error code success-0  fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def Circle(self, desc_pos_p, tool_p, user_p, desc_pos_t, tool_t, user_t, joint_pos_p=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               joint_pos_t=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               vel_p=20.0, acc_p=0.0, exaxis_pos_p=[0.0, 0.0, 0.0, 0.0], vel_t=20.0, acc_t=0.0,
               exaxis_pos_t=[0.0, 0.0, 0.0, 0.0],
               ovl=100.0, offset_flag=0, offset_pos=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]):
        if self.GetSafetyCode() != 0:
            return self.GetSafetyCode()
        desc_pos_p = list(map(float, desc_pos_p))
        tool_p = float(int(tool_p))
        user_p = float(int(user_p))
        joint_pos_p = list(map(float, joint_pos_p))
        vel_p = float(vel_p)
        acc_p = float(acc_p)
        exaxis_pos_p = list(map(float, exaxis_pos_p))

        desc_pos_t = list(map(float, desc_pos_t))
        tool_t = float(int(tool_t))
        user_t = float(int(user_t))
        joint_pos_t = list(map(float, joint_pos_t))
        vel_t = float(vel_t)
        acc_t = float(acc_t)
        exaxis_pos_t = list(map(float, exaxis_pos_t))

        ovl = float(ovl)
        offset_flag = int(offset_flag)
        offset_pos = list(map(float, offset_pos))

        if ((joint_pos_p[0] == 0.0) and (joint_pos_p[1] == 0.0) and (joint_pos_p[2] == 0.0) and (joint_pos_p[3] == 0.0)
                and (joint_pos_p[4] == 0.0) and (joint_pos_p[5] == 0.0)):  # If no parameters are entered, inverse kinematics is called to solve the problem.
            retp = self.robot.GetInverseKin(0, desc_pos_p, -1)  # Resolving inverse kinematics
            if retp[0] == 0:
                joint_pos_p = [retp[1], retp[2], retp[3], retp[4], retp[5], retp[6]]
            else:
                error = retp[0]
                return error

        if ((joint_pos_t[0] == 0.0) and (joint_pos_t[1] == 0.0) and (joint_pos_t[2] == 0.0) and (joint_pos_t[3] == 0.0)
                and (joint_pos_t[4] == 0.0) and (joint_pos_t[5] == 0.0)):  # If no parameters are entered, inverse kinematics is called to solve the problem.
            rett = self.robot.GetInverseKin(0, desc_pos_t, -1)  # Resolving inverse kinematics
            if rett[0] == 0:
                joint_pos_t = [rett[1], rett[2], rett[3], rett[4], rett[5], rett[6]]
            else:
                error = rett[0]
                return error

        error = self.robot.Circle(joint_pos_p, desc_pos_p, [tool_p, user_p, vel_p, acc_p], exaxis_pos_p, joint_pos_t,
                                  desc_pos_t,
                                  [tool_t, user_t, vel_t, acc_t], exaxis_pos_t, ovl, offset_flag, offset_pos)
        return error

    """   
    @brief  Cartesian space spiral motion
    @param  [in] Required parameters desc_pos: Target Cartesian pose，unit [mm][°]
    @param  [in] Required parameters tool: Tool number，[0~14]
    @param  [in] Required parameters user: Workpiece number，[0~14]
    @param  [in] Required parameters param:[circle_num, circle_angle, rad_init, rad_add, rotaxis_add, rot_direction]circle_num: Number of spiral circles，circle_angle: Helical inclination，
    rad_init: Initial radius of spiral，rad_add: Radius increment，rotaxis_add: Rotating axis direction increment，rot_direction: Direction of rotation，0-Clockwise，1-Counterclockwise
    @param  [in] Default parameters joint_pos: Target joint position，unit [°] The default initial value is[0.0,0.0,0.0,0.0,0.0,0.0]，The default value calls inverse kinematics solution to return value
    @param  [in] Default parameters vel：Speed ​​percentage，[0~100] default20.0
    @param  [in] Default parameters acc：Acceleration percentage，[0~100] default100.0
    @param  [in] Default parameters exaxis_pos: External axis 1 Location ~ External axis 4 Location default[0.0,0.0,0.0,0.0]
    @param  [in] Default parameters ovl: Speed ​​scaling factor，[0~100] default100.0
    @param  [in] Default parameters offset_flag:[0]-No offset，[1]-Workpieces/Offset under base coordinate system，[2]-Offset under tool coordinate system default 0
    @param  [in] Default parameters offset_pos: Pose offset，unit [mm][°] default[0.0,0.0,0.0,0.0,0.0,0.0]
    @return Error code success-0  fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def NewSpiral(self, desc_pos, tool, user, param, joint_pos=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], vel=20.0, acc=0.0,
                  exaxis_pos=[0.0, 0.0, 0.0, 0.0],
                  ovl=100.0, offset_flag=0, offset_pos=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]):
        if self.GetSafetyCode() != 0:
            return self.GetSafetyCode()
        desc_pos = list(map(float, desc_pos))
        tool = int(tool)
        user = int(user)
        param[0] = float(param[0])
        param[1] = float(param[1])
        param[2] = float(param[2])
        param[3] = float(param[3])
        param[4] = float(param[4])
        param[5] = float(param[5])
        joint_pos = list(map(float, joint_pos))
        vel = float(vel)
        acc = float(acc)
        exaxis_pos = list(map(float, exaxis_pos))
        ovl = float(ovl)
        offset_flag = int(offset_flag)
        offset_pos = list(map(float, offset_pos))

        if ((joint_pos[0] == 0.0) and (joint_pos[1] == 0.0) and (joint_pos[2] == 0.0) and (joint_pos[3] == 0.0)
                and (joint_pos[4] == 0.0) and (joint_pos[5] == 0.0)):  # If no parameters are entered, inverse kinematics is called to solve the problem.
            ret = self.robot.GetInverseKin(0, desc_pos, -1)  # Resolving inverse kinematics
            if ret[0] == 0:
                joint_pos = [ret[1], ret[2], ret[3], ret[4], ret[5], ret[6]]
            else:
                error = ret[0]
                return error
        error = self.robot.NewSpiral(joint_pos, desc_pos, tool, user, vel, acc, exaxis_pos, ovl, offset_flag,
                                     offset_pos, param)
        return error

    """   
    @brief  Servo movement begins，CooperateServoJ、ServoCartCommand usage
    @param  [in] NULL
    @return Error code success-0  fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def ServoMoveStart(self):
        error = self.robot.ServoMoveStart()
        return error

    """   
    @brief  The servo movement ends，CooperateServoJ、ServoCartCommand usage
    @param  [in] NULL
    @return Error code success-0  fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def ServoMoveEnd(self):
        error = self.robot.ServoMoveEnd()
        return error

    """   
    @brief  Joint space servo mode motion
    @param  [in] Required parameters joint_pos: Target joint position，unit [°]
    @param  [in] Required parameters axisPos  External axis position,unitmm
    @param  [in] Default parameters acc: Acceleration，scope [0~100]，Not open yet，Default is 0.0
    @param  [in] Default parameters vel: speed，scope [0~100]，Not open yet，Default is 0.0
    @param  [in] Default parameters cmdT: Instruction issuance cycle，units，Suggested scope[0.001~0.0016], Default is0.008
    @param  [in] Default parameters filterT: Filtering time，unit [s]，Not open yet， Default is0.0
    @param  [in] Default parameters gain: The proportional amplifier at the target position，Not open yet， Default is0.0
    @return Error code success-0  fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def ServoJ(self, joint_pos,axisPos, acc=0.0, vel=0.0, cmdT=0.008, filterT=0.0, gain=0.0):
        if self.GetSafetyCode() != 0:
            return self.GetSafetyCode()
        joint_pos = list(map(float, joint_pos))
        axisPos = list(map(float, axisPos))
        acc = float(acc)
        vel = float(vel)
        cmdT = float(cmdT)
        filterT = float(filterT)
        gain = float(gain)
        error = self.robot.ServoJ(joint_pos,axisPos,acc, vel, cmdT, filterT, gain)
        return error

    """   
    @brief  Cartesian space servo mode motion
    @param  [in] Required parameters mode:[0]-Absolute exercise (Base coordinate system)，[1]-Incremental motion (Base coordinate system)，[2]-Incremental motion(Tool coordinate system)
    @param  [in] Required parameters desc_pos: Target Cartesian position/Target Cartesian position increment
    @param  [in] Default parameters pos_gain: Position incremental proportional coefficient，Effective only in incremental exercise，scope [0~1], Default is [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    @param  [in] Default parameters acc: Acceleration，scope [0~100]，Not open yet，Default is 0.0
    @param  [in] Default parameters vel: speed，scope [0~100]，Not open yet，Default is 0.0
    @param  [in] Default parameters cmdT:Instruction issuance cycle，units，Suggested scope[0.001~0.0016], Default is0.008
    @param  [in] Default parameters filterT: Filtering time，unit [s]，Not open yet， Default is0.0
    @param  [in] Default parameters gain: The proportional amplifier at the target position，Not open yet， Default is0.0
    @return Error code success-0  fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def ServoCart(self, mode, desc_pos, pos_gain=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], acc=0.0, vel=0.0, cmdT=0.008,
                  filterT=0.0, gain=0.0):
        if self.GetSafetyCode() != 0:
            return self.GetSafetyCode()
        mode = int(mode)
        desc_pos = list(map(float, desc_pos))
        pos_gain = list(map(float, pos_gain))
        acc = float(acc)
        vel = float(vel)
        cmdT = float(cmdT)
        filterT = float(filterT)
        gain = float(gain)
        error = self.robot.ServoCart(mode, desc_pos, pos_gain, acc, vel, cmdT, filterT, gain)
        return error

    """   
    @brief  Joint torque control begins
    @param  [in] NULL
    @return Error code success-0  fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def ServoJTStart(self):
        error = self.robot.ServoJTStart()
        return error

    """   
    @brief  Joint torque control
    @param  [in] Required parameters torque j1~j6Joint torque，unitNm
    @param  [in] Required parameters interval Instruction cycle，units，scope[0.001~0.008]
    @return Error code success-0  fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def ServoJT(self, torque, interval):
        if self.GetSafetyCode() != 0:
            return self.GetSafetyCode()
        torque = list(map(float, torque))
        interval = float(interval)
        error = self.robot.ServoJT(torque, interval)
        return error

    """   
    @brief  Joint torque control ends
    @param  [in] NULL
    @return Error code success-0  fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def ServoJTEnd(self):
        error = self.robot.ServoJTEnd()
        return error

    """   
    @brief  Cartesian space point-to-point motion
    @param  [in] Required parameters desc_pos: Target Cartesian position/Target Cartesian position increment
    @param  [in] Required parameters tool: Tool number，[0~14]
    @param  [in] Required parameters user: Workpiece number，[0~14]
    @param  [in] Default parameters vel: speed，scope [0~100]，Default is 20.0
    @param  [in] Default parameters acc: Acceleration，scope [0~100]，Not open yet,Default is 0.0
    @param  [in] Default parameters ovl: Speed ​​scaling factor，[0~100，Default is 100.0
    @param  [in] Default parameters blendT:[-1.0]-Exercise in place (block)，[0~500]-Smoothing time (Non-blocking)，unit [ms] Default is -1.0
    @param  [in] Default parameters config: Joint configuration，[-1]-Refer to the current joint position to solve，[0~7]-Solve according to joint configuration Default is -1
    @return Error code success-0  fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def MoveCart(self, desc_pos, tool, user, vel=20.0, acc=0.0, ovl=100.0, blendT=-1.0, config=-1):
        if self.GetSafetyCode() != 0:
            return self.GetSafetyCode()
        desc_pos = list(map(float, desc_pos))
        tool = int(tool)
        user = int(user)
        vel = float(vel)
        acc = float(acc)
        ovl = float(ovl)
        blendT = float(blendT)
        config = int(config)
        error = self.robot.MoveCart(desc_pos, tool, user, vel, acc, ovl, blendT, config)
        return error

    """   
    @brief  Spline movement begins
    @param  [in] NULL
    @return Error code success-0  fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def SplineStart(self):
        error = self.robot.SplineStart()
        return error

    """   
    @brief  Spline movement PTP
    @param  [in] Required parameters joint_pos: Target joint position，unit [°]
    @param  [in] Required parameters tool: Tool number，[0~14]
    @param  [in] Required parameters user: Workpiece number，[0~14]
    @param  [in] Default parameters desc_pos: Target Cartesian pose，unit [mm][°] The default initial value is[0.0,0.0,0.0,0.0,0.0,0.0]，The default value calls positive kinematics solution to return value
    @param  [in] Default parameters vel: speed，scope [0~100]，Default is 20.0
    @param  [in] Default parameters acc: Acceleration，scope [0~100]，Default is 100.0
    @param  [in] Default parameters ovl: Speed ​​scaling factor，[0~100，Default is 100.0
    @return Error code success-0  fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def SplinePTP(self, joint_pos, tool, user, desc_pos=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], vel=20.0, acc=100.0, ovl=100.0):
        if self.GetSafetyCode() != 0:
            return self.GetSafetyCode()
        joint_pos = list(map(float, joint_pos))
        tool = int(tool)
        user = int(user)
        desc_pos = list(map(float, desc_pos))
        vel = float(vel)
        acc = float(acc)
        ovl = float(ovl)
        if ((desc_pos[0] == 0.0) and (desc_pos[1] == 0.0) and (desc_pos[2] == 0.0) and (desc_pos[3] == 0.0)
                and (desc_pos[4] == 0.0) and (desc_pos[5] == 0.0)):  # If no parameters are entered, call positive kinematics to solve
            ret = self.robot.GetForwardKin(joint_pos)  # Solve the kinematics
            if ret[0] == 0:
                desc_pos = [ret[1], ret[2], ret[3], ret[4], ret[5], ret[6]]
            else:
                error = ret[0]
                return error
        error = self.robot.SplinePTP(joint_pos, desc_pos, tool, user, vel, acc, ovl)
        return error

    """   
    @brief  Spline movement ends
    @param  [in] NULL
    @return Error code success-0  fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def SplineEnd(self):
        error = self.robot.SplineEnd()
        return error

    """   
    @brief  A new spline movement begins
    @param  [in] Required parameters type:0-Arc transition，1-The path point is given
    @param  [in] Default parameters averageTime: Global average connection time（ms）Default is 2000
    @return Error code success-0  fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def NewSplineStart(self, type, averageTime=2000):
        type = int(type)
        averageTime = int(averageTime)
        error = self.robot.NewSplineStart(type, averageTime)
        return error

    """   
    @brief  New spline command point
    @param  [in] Required parameters desc_pos: Target Cartesian pose，unit [mm][°]
    @param  [in] Required parameters tool: Tool number，[0~14]
    @param  [in] Required parameters user: Workpiece number，[0~14]
    @param  [in] Required parameters lastFlag: Is it the last point，0-no，1-yes
    @param  [in] Default parameters joint_pos: Target joint position，unit [°] The default initial value is[0.0,0.0,0.0,0.0,0.0,0.0]，The default value calls inverse kinematics solution to return value
    @param  [in] Default parameters vel: speed，scope [0~100]，Not open yet，Default is 0.0
    @param  [in] Default parameters acc: Acceleration，scope [0~100]，Not open yet，Default is 0.0
    @param  [in] Default parameters ovl: Speed ​​scaling factor，[0~100] Default is 100.0
    @param  [in] Default parameters blendR: [0~1000]-Smooth radius，unit [mm] default0.0
    @return Error code success-0  fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def NewSplinePoint(self, desc_pos, tool, user, lastFlag, joint_pos=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], vel=0.0,
                       acc=0.0, ovl=100.0, blendR=0.0):
        if self.GetSafetyCode() != 0:
            return self.GetSafetyCode()
        desc_pos = list(map(float, desc_pos))
        tool = int(tool)
        user = int(user)
        lastFlag = int(lastFlag)
        joint_pos = list(map(float, joint_pos))
        vel = float(vel)
        acc = float(acc)
        ovl = float(ovl)
        blendR = float(blendR)
        if ((joint_pos[0] == 0.0) and (joint_pos[1] == 0.0) and (joint_pos[2] == 0.0) and (joint_pos[3] == 0.0)
                and (joint_pos[4] == 0.0) and (joint_pos[5] == 0.0)):  # If no parameters are entered, inverse kinematics is called to solve the problem.
            ret = self.robot.GetInverseKin(0, desc_pos, -1)  # Resolving inverse kinematics
            if ret[0] == 0:
                joint_pos = [ret[1], ret[2], ret[3], ret[4], ret[5], ret[6]]
            else:
                error = ret[0]
                return error
        error = self.robot.NewSplinePoint(joint_pos, desc_pos, tool, user, vel, acc, ovl, blendR, lastFlag)
        return error

    """   
    @brief  The new spline movement ends
    @param  [in] NULL
    @return Error code success-0  fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def NewSplineEnd(self):
        error = self.robot.NewSplineEnd()
        return error

    """   
    @brief  End the movement
    @param  [in] NULL
    @return Error code success-0  fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def StopMotion(self):
        error = self.robot.StopMotion()
        return error

    """   
    @brief  Pause exercise
    @param  [in] NULL
    @return Error code success-0  fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def PauseMotion(self):
        # error = self.robot.PauseMotion()
        self.send_message("/f/bIII0III103III5IIIPAUSEIII/b/f")
        return 0

        # return error

    """   
    @brief  Resuming exercise
    @param  [in] NULL
    @return Error code success-0  fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def ResumeMotion(self):
        if self.GetSafetyCode() != 0:
            return self.GetSafetyCode()
        # error = self.robot.ResumeMotion()
        error = self.send_message("/f/bIII0III104III6IIIRESUMEIII/b/f")
        return error

    """   
    @brief  The overall offset of the point begins
    @param  [in] Required parameters flag:0-Offset under base coordinate or workpiece coordinate system，2-Offset under tool coordinate system
    @param  [in] Required parameters offset_pos: Offset，unit [mm][°]。
    @return Error code success-0  fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def PointsOffsetEnable(self, flag, offset_pos):
        flag = int(flag)
        offset_pos = list(map(float, offset_pos))
        error = self.robot.PointsOffsetEnable(flag, offset_pos)
        return error

    """   
    @brief  The overall offset of the point ends
    @param  [in] NULL
    @return Error code success-0  fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def PointsOffsetDisable(self):
        error = self.robot.PointsOffsetDisable()
        return error

    """   
    ***************************************************************************robotIO********************************************************************************************
    """

    """   
    @brief  Set the digital output of the control box
    @param  [in] Required parameters id:io serial number，scope [0~15]
    @param  [in] Required parameters status:0-close，1-open
    @param  [in] Default parameters smooth:0-Non-smooth，1-smooth default0
    @param  [in] Default parameters block:0-block，1-Non-blocking default0
    @return Error code success-0  fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def SetDO(self, id, status, smooth=0, block=0):
        id = int(id)
        status = int(status)
        smooth = int(smooth)
        block = int(block)
        error = self.robot.SetDO(id, status, smooth, block)
        return error

    """   
    @brief  Set the digital output of the tool
    @param  [in] Required parameters id:io serial number，scope [0~1]
    @param  [in] Required parameters status:0-close，1-open
    @param  [in] Default parameters smooth:0-Non-smooth，1-smooth default0
    @param  [in] Default parameters block:0-block，1-Non-blocking default0
    @return Error code success-0  fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def SetToolDO(self, id, status, smooth=0, block=0):
        id = int(id)
        status = int(status)
        smooth = int(smooth)
        block = int(block)
        error = self.robot.SetToolDO(id, status, smooth, block)
        return error

    """   
    @brief  Set the analog output of the control box
    @param  [in] Required parameters id:io serial number，scope [0~1]
    @param  [in] Required parameters value: Current or voltage percentage，scope [0~100%] Corresponding current value [0~20mA] or voltage [0~10V]；
    @param  [in] Default parameters block:0-block，1-Non-blocking default0
    @return Error code success-0  fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def SetAO(self, id, value, block=0):
        id = int(id)
        value = float(value)
        block = int(block)
        error = self.robot.SetAO(id, value * 40.95, block)
        return error

    """   
    @brief  Set the tool analog output
    @param  [in] Required parameters id:io serial number，scope [0]
    @param  [in] Required parameters value: Current or voltage percentage，scope [0~100%] Corresponding current value [0~20mA] or voltage [0~10V]；
    @param  [in] Default parameters block:0-block，1-Non-blocking default0
    @return Error code success-0  fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def SetToolAO(self, id, value, block=0):
        id = int(id)
        value = float(value)
        block = int(block)
        error = self.robot.SetToolAO(id, value * 40.95, block)
        return error

    """   
    @brief  Get the digital input of the control box
    @param  [in] Required parameters id:io serial number，scope [0-15]
    @param  [in] Default parameters block:0-block，1-Non-blocking default0
    @return Error code success-0  fail-Error code
    @return Return value（The call returns successfully）di: 0-Low level，1-High level
    """

    @log_call
    # @xmlrpc_timeout
    def GetDI(self, id, block=0):
        id = int(id)
        block = int(block)
        # _error = self.robot.GetDI(id, block)
        # error = _error[0]
        # print(_error)
        # if _error[0] == 0:
        #     di = _error[1]
        #     return error, di
        # else:
        #     return error
        if 0 <= id < 8:
            level = (self.robot_state_pkg.cl_dgt_input_l & (0x01 << id)) >> id
            return 0, level
        elif 8 <= id < 16:
            id -= 8
            level = (self.robot_state_pkg.cl_dgt_input_h & (0x01 << id)) >> id
            return 0, level
        else:
            return -1


    """   
    @brief  Get the digital input of the tool
    @param  [in] Required parameters id:io serial number，scope [0~1]
    @param  [in] Default parameters block:0-block，1-Non-blocking default0
    @return Error code success-0,  fail-Error code
    @return Return value（The call returns successfully）di: 0-Low level，1-High level
    """

    @log_call
    # @xmlrpc_timeout
    def GetToolDI(self, id, block=0):
        id = int(id)
        block = int(block)
        # _error = self.robot.GetToolDI(id, block)
        # error = _error[0]
        # if _error[0] == 0:
        #     di = _error[1]
        #     return error, di
        # else:
        #     return error
        if 0 <= id < 2:
            id+=1
            level = (self.robot_state_pkg.tl_dgt_input_l & (0x01 << id)) >> id
            return 0,level
        else:
            return -1

    """   
    @brief  Wait for the digital input of the control box
    @param  [in] Required parameters id:io serial number，scope [0~15]
    @param  [in] Required parameters status:0-close，1-open
    @param  [in] Required parameters maxtime: Maximum waiting time，unit [ms]
    @param  [in] Required parameters opt: Post-timeout strategy，0-The program stops and prompts timeout，1-Ignore the timeout prompt to continue execution，2-Waiting forever
    @return Error code success-0  fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def WaitDI(self, id, status, maxtime, opt):
        id = int(id)
        status = int(status)
        maxtime = int(maxtime)
        opt = int(opt)
        error = self.robot.WaitDI(id, status, maxtime, opt)
        return error

    """   
    @brief  Wait for the control box multi-channel digital input
    @param  [in] Required parameters mode 0-Multiple channels and，1-Multiple or
    @param  [in] Required parameters id  ioserial number，bit0~bit7correspondDI0~DI7，bit8~bit15correspondCI0~CI7
    @param  [in] Required parameters status:0-close，1-open
    @param  [in] Required parameters maxtime: Maximum waiting time，unit [ms]
    @param  [in] Required parameters opt: 超时后策略，0-The program stops and prompts timeout，1-Ignore the timeout prompt to continue execution，2-Waiting forever
    @return Error code success-0  fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def WaitMultiDI(self, mode, id, status, maxtime, opt):
        mode = int(mode)
        id = int(id)
        status = int(status)
        maxtime = int(maxtime)
        opt = int(opt)
        error = self.robot.WaitMultiDI(mode, id, status, maxtime, opt)
        return error

    """   
    @brief  Wait for the digital input of the tool
    @param  [in] Required parameters id:io serial number，scope [0~1]
    @param  [in] Required parameters status:0-close，1-open
    @param  [in] Required parameters maxtime: Maximum waiting time，unit [ms]
    @param  [in] Required parameters opt: Post-timeout strategy，0-The program stops and prompts timeout，1-Ignore the timeout prompt to continue execution，2-Waiting forever
    @return Error code success-0  fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def WaitToolDI(self, id, status, maxtime, opt):
        id = int(id)
        id = id+1 #Inside the controller1corresponddi0,2corresponddi1
        status = int(status)
        maxtime = int(maxtime)
        opt = int(opt)
        error = self.robot.WaitToolDI(id, status, maxtime, opt)
        return error

    """   
    @brief  Obtain the analog input of the control box
    @param  [in] Required parameters id:io serial number，scope [0~1]
    @param  [in] Default parameters block:0-阻塞，1-Non-blocking default0
    @return Error code success-0,  fail-Error code
    @return Return value（The call returns successfully）value: Input current or voltage percentage，scope [0~100] Corresponding current value [0~20mA] or voltage [0~10V]
    """

    @log_call
    @xmlrpc_timeout
    def GetAI(self, id, block=0):
        id = int(id)
        block = int(block)
        # _error = self.robot.GetAI(id, block)
        # error = _error[0]
        # if _error[0] == 0:
        #     value = _error[1]
        #     return error, value
        # else:
        #     return error
        if 0 <= id < 2:
            return 0,self.robot_state_pkg.cl_analog_input[id] / 40.95
        else:
            return -1


    """   
    @brief  Get the tool analog input
    @param  [in] Required parameters id:io serial number，scope [0]
    @param  [in] Default parameters block:0-block，1-Non-blocking default0
    @return Error code success-0,  fail-Error code
    @return Return value（The call returns successfully）value: Input current or voltage percentage，scope [0~100] Corresponding current value [0~20mA] or voltage [0~10V]
    """

    @log_call
    @xmlrpc_timeout
    def GetToolAI(self, id, block=0):
        id = int(id)
        block = int(block)
        # _error = self.robot.GetToolAI(id, block)
        # error = _error[0]
        # if _error[0] == 0:
        #     value = _error[1]
        #     return error, value
        # else:
        #     return error
        return 0, self.robot_state_pkg.tl_anglog_input / 40.95

    """   
    @brief  Get the status of the end point of the robot record button
    @param  [in] NULL
    @return Error code success-0,  fail-Error code
    @return Return value（The call returns successfully）Button Status，0-Press，1-release
    """

    @log_call
    @xmlrpc_timeout
    def GetAxlePointRecordBtnState(self):
        _error = self.robot.GetAxlePointRecordBtnState()
        error = _error[0]
        if _error[0] == 0:
            value = _error[1]
            return error, value
        else:
            return error


    """   
    @brief  Get the end of the robotDOOutput status
    @param  [in] NULL
    @return Error code success-0,  fail-Error code
    @return Return value（The call returns successfully）do_state DOOutput status，do0~do1correspondbit1~bit2,frombit0start
    """

    @log_call
    @xmlrpc_timeout
    def GetToolDO(self):
        # _error = self.robot.GetToolDO()
        # error = _error[0]
        # if _error[0] == 0:
        #     value = _error[1]
        #     return error, value
        # else:
        #     return error
        return 0,self.robot_state_pkg.tl_dgt_output_l

    """   
    @brief  Get the robot controllerDOOutput status
    @param  [in] NULL
    @return Error code success-0,  fail-Error code
    @return Return value（The call returns successfully）do_state_h DOOutput status，co0~co7correspondbit0~bit7 do_state_l DOOutput status，do0~do7correspondbit0~bit7
    """

    @log_call
    @xmlrpc_timeout
    def GetDO(self):
        # _error = self.robot.GetDO()
        # error = _error[0]
        # if _error[0] == 0:
        #     do_state_h = _error[1]
        #     do_state_l = _error[2]
        #     return error, [do_state_h, do_state_l]
        # else:
        #     return error
        return 0, [self.robot_state_pkg.cl_dgt_output_h,self.robot_state_pkg.cl_dgt_output_l]

    """   
    @brief  Wait for the control box analog input
    @param  [in] Required parameters id:io serial number，scope [0~1]
    @param  [in] Required parameters sign:0-Greater than，1-Less than
    @param  [in] Required parameters value: Input current or voltage percentage，scope [0~100] Corresponding current value [0~20mA] or voltage [0~10V]
    @param  [in] Required parameters maxtime: Maximum waiting time，unit [ms]
    @param  [in] Required parameters opt: Post-timeout strategy，0-The program stops and prompts timeout，1-Ignore the timeout prompt to continue execution，2-Waiting forever
    @return Error code success-0  fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def WaitAI(self, id, sign, value, maxtime, opt):
        id = int(id)
        sign = int(sign)
        value = float(value)
        maxtime = int(maxtime)
        opt = int(opt)
        error = self.robot.WaitAI(id, sign, value*40.95, maxtime, opt)
        return error

    """   
    @brief  Wait for the tool analog input
    @param  [in] Required parameters id:io serial number，scope [0]
    @param  [in] Required parameters sign:0-Greater than，1-Less than
    @param  [in] Required parameters value: Input current or voltage percentage，scope [0~100] Corresponding current value [0~20mA] or voltage [0~10V]
    @param  [in] Required parameters maxtime: Maximum waiting time，unit [ms]
    @param  [in] Required parameters opt: Post-timeout strategy，0-The program stops and prompts timeout，1-Ignore the timeout prompt to continue execution，2-Waiting forever
    @return Error code success-0  fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def WaitToolAI(self, id, sign, value, maxtime, opt):
        id = int(id)
        sign = int(sign)
        value = float(value)
        maxtime = int(maxtime)
        opt = int(opt)
        error = self.robot.WaitToolAI(id, sign, value*40.95, maxtime, opt)
        return error

    """   
    ***************************************************************************Common settings for robots********************************************************************************************
    """

    """   
    @brief  Set global speed
    @param  [in] Required parameters vel  Speed ​​percentage，scope[0~100]
    @return Error code success-0  fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def SetSpeed(self, vel):
        vel = int(vel)
        error = self.robot.SetSpeed(vel)
        return error

    """   
    @brief  Set system variables
    @param  [in] Required parameters id：Variable number，scope [1~20]
    @param  [in] Required parameters value：Variable value
    @return Error code success-0  fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def SetSysVarValue(self, id, value):
        id = int(id)
        value = float(value)
        error = self.robot.SetSysVarValue(id, value)
        return error

    """   
    @brief  Set tool reference points-Six-point method
    @param  [in] Required parameters point_num Point number,scope[1~6] 
    @return Error code success-0  fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def SetToolPoint(self, point_num):
        point_num = int(point_num)
        error = self.robot.SetToolPoint(point_num)
        return error

    """   
    @brief  Calculate tool coordinate system-Six-point method
    @param  [in] NULL
    @return Error code success-0,  fail-Error code
    @return Return value（The call returns successfully）tcp_pose [x,y,z,rx,ry,rz] Tool coordinate system
    """

    @log_call
    @xmlrpc_timeout
    def ComputeTool(self):
        _error = self.robot.ComputeTool()
        error = _error[0]
        if _error[0] == 0:
            return error, [_error[1], _error[2], _error[3], _error[4], _error[5], _error[6]]
        else:
            return error

    """   
    @brief  Set tool reference points-Four-point method
    @param  [in] Required parameters point_num Point number,scope[1~4] 
    @return Error code success-0  fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def SetTcp4RefPoint(self, point_num):
        point_num = int(point_num)
        error = self.robot.SetTcp4RefPoint(point_num)
        return error

    """   
    @brief  Calculate tool coordinate system-Four-point method
    @param  [in] NULL
    @return Error code success-0,  fail-Error code
    @return Return value（The call returns successfully）tcp_pose [x,y,z,rx,ry,rz]  Tool coordinate system
    """

    @log_call
    @xmlrpc_timeout
    def ComputeTcp4(self):
        _error = self.robot.ComputeTcp4()
        error = _error[0]
        if _error[0] == 0:
            return error, [_error[1], _error[2], _error[3], _error[4], _error[5], _error[6]]
        else:
            return error

    """   
    @brief  Set the tool coordinate system
    @param  [in] Required parameters id: Coordinate system number，scope [1~15]
    @param  [in] Required parameters t_coord:[x,y,z,rx,ry,rz]  Position of the center point of the tool relative to the end flange，unit [mm][°]
    @param  [in] Required parameters type:0-Tool coordinate system，1-Sensor coordinate system
    @param  [in] Required parameters install: Installation location，0-Robot end，1-External robot
    @param  [in] Required parameters toolID: toolID
    @param  [in] Required parameters loadNum: Load number
    @return Error code success-0  fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def SetToolCoord(self, id, t_coord, type, install, toolID, loadNum):
        id = int(id)
        t_coord = list(map(float, t_coord))
        type = int(type)
        install = int(install)
        toolID = int(toolID)
        loadNum = int(loadNum)
        error = self.robot.SetToolCoord(id, t_coord, type, install, toolID, loadNum)
        return error

    """   
    @brief  Set the tool coordinate series table
    @param  [in] Required parameters id: Coordinate system number，scope [1~15]
    @param  [in] Required parameters t_coord:[x,y,z,rx,ry,rz]  Position of the center point of the tool relative to the end flange，unit [mm][°]
    @param  [in] Required parameters type:0-Tool coordinate system，1-Sensor coordinate system
    @param  [in] Required parameters install: Installation location，0-Robot end，1-External robot
    @param  [in] Required parameters loadNum: Load number
    @return Error code success-0  fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def SetToolList(self, id, t_coord, type, install , loadNum):
        id = int(id)
        t_coord = list(map(float, t_coord))
        type = int(type)
        install = int(install)
        loadNum = int(loadNum)
        error = self.robot.SetToolList(id, t_coord, type, install, loadNum)
        return error

    """   
    @brief  Set external tool reference points-Three-point method
    @param  [in] Required parameters point_num Point number,scope[1~3] 
    @return Error code success-0  fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def SetExTCPPoint(self, point_num):
        point_num = int(point_num)
        error = self.robot.SetExTCPPoint(point_num)
        return error

    """   
    @brief  Calculate external tool coordinate system-Three-point method
    @param  [in] NULL
    @return Error code success-0,  fail-Error code
    @return Return value（The call returns successfully）tcp_pose [x,y,z,rx,ry,rz] External tool coordinate system
    """

    @log_call
    @xmlrpc_timeout
    def ComputeExTCF(self):
        _error = self.robot.ComputeExTCF()
        error = _error[0]
        if _error[0] == 0:
            return error, [_error[1], _error[2], _error[3], _error[4], _error[5], _error[6]]
        else:
            return error

    """   
    @brief  Set external tool coordinate system
    @param  [in] Required parameters id: Coordinate system number，scope [0~14]
    @param  [in] Required parameters etcp: [x,y,z,rx,ry,rz] External tool coordinate system，unit [mm][°]
    @param  [in] Required parameters etool: [x,y,z,rx,ry,rz] End tool coordinate system，unit [mm][°]
    @return Error code success-0  fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def SetExToolCoord(self, id, etcp, etool):
        id = int(id)
        etcp = list(map(float, etcp))
        etool = list(map(float, etool))
        error = self.robot.SetExToolCoord(id, etcp, etool)
        return error

    """   
    @brief  Set external tool coordinate series table
    @param  [in] Required parameters id: Coordinate system number，scope [0~14]
    @param  [in] Required parameters etcp: [x,y,z,rx,ry,rz] External tool coordinate system，unit [mm][°]
    @param  [in] Required parameters etool: [x,y,z,rx,ry,rz] End tool coordinate system，unit [mm][°]
    @return Error code success-0  fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def SetExToolList(self, id, etcp, etool):
        id = int(id)
        etcp = list(map(float, etcp))
        etool = list(map(float, etool))
        error = self.robot.SetExToolList(id, etcp, etool)
        return error

    """   
    @brief  Set the workpiece reference point-Three-point method
    @param  [in] Required parameters point_num Point number,scope[1~3] 
    @return Error code success-0  fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def SetWObjCoordPoint(self, point_num):
        point_num = int(point_num)
        error = self.robot.SetWObjCoordPoint(point_num)
        return error

    """   
    @brief  Calculate the workpiece coordinate system
    @param  [in] method Calculation method 0：origin-xaxis-zaxis  1：origin-xaxis-xyflat
    @param  [in] refFrame Reference coordinate system
    @return Error code success-0,  fail-Error code
    @return Return value（The call returns successfully）wobj_pose [x,y,z,rx,ry,rz] Workpiece coordinate system
    """

    @log_call
    @xmlrpc_timeout
    def ComputeWObjCoord(self, method, refFrame):
        method = int(method)
        refFrame = int(refFrame)
        _error = self.robot.ComputeWObjCoord(method, refFrame)
        error = _error[0]
        if _error[0] == 0:
            return error, [_error[1], _error[2], _error[3], _error[4], _error[5], _error[6]]
        else:
            return error

    """   
    @brief  Set the workpiece coordinate system
    @param  [in] Required parameters id: Coordinate system number，scope [0~14]
    @param  [in] Required parameters coord: The workpiece coordinate system relative to the center position of the end flange，unit [mm][°]
    @param  [in] Required parameters refFrame: Reference coordinate system
    @return Error code success-0  fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def SetWObjCoord(self, id, coord, refFrame):
        id = int(id)
        coord = list(map(float, coord))
        refFrame = int(refFrame)
        error = self.robot.SetWObjCoord(id, coord, refFrame)
        return error

    """   
    @brief  Set the workpiece coordinate series table
    @param  [in] Required parameters id: Coordinate system number，scope [0~14]
    @param  [in] Required parameters coord: The workpiece coordinate system relative to the center position of the end flange，unit [mm][°]
    @param  [in] Required parameters refFrame: Reference coordinate system
    @return Error code success-0  fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def SetWObjList(self, id, coord, refFrame):
        id = int(id)
        coord = list(map(float, coord))
        refFrame = int(refFrame)
        error = self.robot.SetWObjList(id, coord, refFrame)
        return error

    """   
    @brief  Set the end load weight
    @param  [in] Required parameters loadNum Load number
    @param  [in] Required parameters weight: unit [kg]
    @return Error code success-0  fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def SetLoadWeight(self, loadNum, weight):
        loadNum = int(loadNum)
        weight = float(weight)
        error = self.robot.SetLoadWeight(loadNum,weight)
        return error

    """   
    @brief  Set up the robot installation method-Fixed installation
    @param  [in] Required parameters method:0-Formal wear，1-Side installation，2-Hang up
    @return Error code success-0  fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def SetRobotInstallPos(self, method):
        method = int(method)
        error = self.robot.SetRobotInstallPos(method)
        return error

    """   
    @brief  Set the robot installation angle
    @param  [in] Required parameters yangle：Incline angle
    @param  [in] Required parameters zangle：Rotating angle
    @return Error code success-0  fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def SetRobotInstallAngle(self, yangle, zangle):
        yangle = float(yangle)
        zangle = float(zangle)
        error = self.robot.SetRobotInstallAngle(yangle, zangle)
        return error

    """   
    @brief  Set the end load centroid coordinates
    @param  [in] Required parameters x: Centroid coordinates，unit [mm]
    @param  [in] Required parameters y: Centroid coordinates，unit [mm]
    @param  [in] Required parameters z: Centroid coordinates，unit [mm]
    @return Error code success-0  fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def SetLoadCoord(self, x, y, z):
        x = float(x)
        y = float(y)
        z = float(z)
        error = self.robot.SetLoadCoord(x, y, z)
        return error

    """   
    @brief  Wait for a specified time
    @param  [in] Required parameters t_ms: unit [ms]
    @return Error code success-0  fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def WaitMs(self, t_ms):
        t_ms = int(t_ms)
        error = self.robot.WaitMs(t_ms)
        return error

    """   
    ***************************************************************************Robot security settings********************************************************************************************
    """

    """   
    @brief  Set collision level
    @param  [in] Required parameters mode:0-grade，1-percentage
    @param  [in] Required parameters level=[j1,j2,j3,j4,j5,j6]: Collision Threshold mode=0hour，scope：1-10 correspondmode=1hour，scope0-100%
    @param  [in] Required parameters config:0-Not updating the configuration file，1-Update configuration files
    @return Error code success-0  fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def SetAnticollision(self, mode, level, config):
        mode = int(mode)
        level = list(map(float, level))
        config = int(config)
        error = self.robot.SetAnticollision(mode, level, config)
        return error

    """   
    @brief  Setting post-collision strategy
    @param  [in] Required parameters strategy：0-Report an error pause，1-Continue to run，2-Error reporting stop，3-Gravity torque mode，4-Oscillation corresponding mode，5-Collision rebound mode
    @param  [in] Default parameters safeTime：Safe stop time[1000-2000]ms，Default is：1000
    @param  [in] Default parameters safeDistance：Safe stop distance[1-150]mm，Default is：100
    @param  [in] Default parameters safeVel：Safe stop speed[50-250]mm/s，Default is：250
    @param  [in] Default parameters safetyMargin[6]：Safety factor[1-10]，Default is：[10,10,10,10,10,10]
    @return Error code success-0  fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def SetCollisionStrategy(self, strategy,safeTime=1000,safeDistance=100,safeVel=250,safetyMargin=[10,10,10,10,10,10]):
        strategy = int(strategy)
        safeTime = int(safeTime)
        safeDistance = int(safeDistance)
        safeVel = int(safeVel)
        safetyMargin = list(map(int, safetyMargin))
        error = self.robot.SetCollisionStrategy(strategy,safeTime,safeDistance,safeVel,safetyMargin)
        return error

    """   
    @brief  Set the positive limit
    @param  [in] Required parameters p_limit=[j1,j2,j3,j4,j5,j6]：Six joint positions
    @return Error code success-0  fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def SetLimitPositive(self, p_limit):
        p_limit = list(map(float, p_limit))
        error = self.robot.SetLimitPositive(p_limit)
        return error

    """   
    @brief  Set negative limit
    @param  [in] Required parameters n_limit=[j1,j2,j3,j4,j5,j6]：Six joint positions
    @return Error code success-0  fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def SetLimitNegative(self, n_limit):
        n_limit = list(map(float, n_limit))
        error = self.robot.SetLimitNegative(n_limit)
        return error

    """   
    @brief  Error status clear，Only resettable errors can be cleared
    @param  [in] NULL
    @return Error code success-0  fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def ResetAllError(self):
        error = self.robot.ResetAllError()
        return error

    """   
    @brief  Joint friction compensation switch
    @param  [in] Required parameters state：0-close，1-open
    @return Error code success-0  fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def FrictionCompensationOnOff(self, state):
        state = int(state)
        error = self.robot.FrictionCompensationOnOff(state)
        return error

    """   
    @brief  Set joint friction compensation coefficient-Fixed installation-Formal wear
    @param  [in] Required parameters coeff=[j1,j2,j3,j4,j5,j6]：Six joint compensation coefficients
    @return Error code success-0  fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def SetFrictionValue_level(self, coeff):
        coeff = list(map(float, coeff))
        error = self.robot.SetFrictionValue_level(coeff)
        return error

    """   
    @brief  Set joint friction compensation coefficient-Fixed installation-Side installation
    @param  [in] Required parameters coeff=[j1,j2,j3,j4,j5,j6]：Six joint compensation coefficients
    @return Error code success-0  fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def SetFrictionValue_wall(self, coeff):
        coeff = list(map(float, coeff))
        error = self.robot.SetFrictionValue_wall(coeff)
        return error

    """   
    @brief  Set joint friction compensation coefficient-Fixed installation-Inverted
    @param  [in] Required parameters coeff=[j1,j2,j3,j4,j5,j6]：Six joint compensation coefficients
    @return Error code success-0  fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def SetFrictionValue_ceiling(self, coeff):
        coeff = list(map(float, coeff))
        error = self.robot.SetFrictionValue_ceiling(coeff)
        return error

    """   
    @brief  Set joint friction compensation coefficient-Free installation
    @param  [in] Required parameters coeff=[j1,j2,j3,j4,j5,j6]：Six joint compensation coefficients
    @return Error code success-0  fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def SetFrictionValue_freedom(self, coeff):
        coeff = list(map(float, coeff))
        error = self.robot.SetFrictionValue_freedom(coeff)
        return error

    """   
    ***************************************************************************Robot status query********************************************************************************************
    """

    """   
    @brief  Get the robot installation angle
    @param  [in] NULL
    @return Error code success- 0,  fail-Error code
    @return Return value（The call returns successfully）[yangle,zangle] yangle-Incline angle,zangle-Rotating angle
    """

    @log_call
    @xmlrpc_timeout
    def GetRobotInstallAngle(self):
        _error = self.robot.GetRobotInstallAngle()
        error = _error[0]
        if error == 0:
            return error, [_error[1], _error[2]]
        else:
            return error

    """   
    @brief  Get system variable values
    @param  [in] id：System variable number，scope [1~20]
    @return Error code success- 0,  fail-Error code
    @return Return value（The call returns successfully） var_value：System variable value
    """

    @log_call
    @xmlrpc_timeout
    def GetSysVarValue(self, id):
        id = int(id)
        _error = self.robot.GetSysVarValue(id)
        error = _error[0]
        if error == 0:
            return error, _error[1]
        else:
            return error

    """   
    @brief  Get the current joint position (angle)
    @param  [in] Default parameters flag：0-block，1-Non-blocking default1
    @return Error code success- 0,  fail-Error code
    @return Return value（The call returns successfully） joint_pos=[j1,j2,j3,j4,j5,j6]
    """

    @log_call
    @xmlrpc_timeout
    def GetActualJointPosDegree(self, flag=1):
        flag = int(flag)
        # _error = self.robot.GetActualJointPosDegree(flag)
        # error = _error[0]
        # if error == 0:
        #     return error, [_error[1], _error[2], _error[3], _error[4], _error[5], _error[6]]
        # else:
        #     return error
        return 0,[self.robot_state_pkg.jt_cur_pos[0],self.robot_state_pkg.jt_cur_pos[1],self.robot_state_pkg.jt_cur_pos[2],
                  self.robot_state_pkg.jt_cur_pos[3],self.robot_state_pkg.jt_cur_pos[4],self.robot_state_pkg.jt_cur_pos[5]]
    """   
    @brief  Get the current position of the joint (radian)
    @param  [in] Default parameters flag：0-block，1-Non-blocking default1
    @return Error code success- 0,  fail-Error code
    @return Return value（The call returns successfully） joint_pos=[j1,j2,j3,j4,j5,j6]
    """

    @log_call
    @xmlrpc_timeout
    def GetActualJointPosRadian(self, flag=1):
        flag = int(flag)
        _error = self.robot.GetActualJointPosRadian(flag)
        error = _error[0]
        if error == 0:
            return error, [_error[1], _error[2], _error[3], _error[4], _error[5], _error[6]]
        else:
            return error

    """   
    @brief  Obtain joint feedback speed-deg/s
    @param  [in] Default parameters flag：0-block，1-Non-blocking default1
    @return Error code success- 0,  fail-Error code
    @return Return value（The call returns successfully） speed=[j1,j2,j3,j4,j5,j6]
    """

    @log_call
    @xmlrpc_timeout
    def GetActualJointSpeedsDegree(self, flag=1):
        flag = int(flag)
        # _error = self.robot.GetActualJointSpeedsDegree(flag)
        # error = _error[0]
        # if error == 0:
        #     return error, [_error[1], _error[2], _error[3], _error[4], _error[5], _error[6]]
        # else:
        #     return error
        return 0,[self.robot_state_pkg.actual_qd[0],self.robot_state_pkg.actual_qd[1],self.robot_state_pkg.actual_qd[2],
                  self.robot_state_pkg.actual_qd[3],self.robot_state_pkg.actual_qd[4],self.robot_state_pkg.actual_qd[5]]

    """   
    @brief  Obtain joint feedback acceleration-deg/s^2
    @param  [in] Default parameters flag：0-block，1-Non-blocking default1
    @return Error code success- 0,  fail-Error code
    @return Return value（The call returns successfully） acc=[j1,j2,j3,j4,j5,j6]
    """

    @log_call
    @xmlrpc_timeout
    def GetActualJointAccDegree(self, flag=1):
        flag = int(flag)
        # _error = self.robot.GetActualJointAccDegree(flag)
        # error = _error[0]
        # if error == 0:
        #     return error, [_error[1], _error[2], _error[3], _error[4], _error[5], _error[6]]
        # else:
        #     return error
        return 0,[self.robot_state_pkg.actual_qdd[0],self.robot_state_pkg.actual_qdd[1],self.robot_state_pkg.actual_qdd[2],
                  self.robot_state_pkg.actual_qdd[3],self.robot_state_pkg.actual_qdd[4],self.robot_state_pkg.actual_qdd[5]]

    """   
    @brief  GetTCPCommand speed
    @param  [in] Default parameters flag：0-block，1-Non-blocking default1
    @return Error code success- 0,  fail-Error code
    @return Return value（The call returns successfully）[tcp_speed,ori_speed] tcp_speed Linear velocity ori_speed Position speed 
    """

    @log_call
    @xmlrpc_timeout
    def GetTargetTCPCompositeSpeed(self, flag=1):
        flag = int(flag)
        # _error = self.robot.GetTargetTCPCompositeSpeed(flag)
        # error = _error[0]
        # if error == 0:
        #     return error, [_error[1], _error[2]]
        # else:
        #     return error
        return 0,[self.robot_state_pkg.target_TCP_CmpSpeed[0],self.robot_state_pkg.target_TCP_CmpSpeed[1]]

    """   
    @brief  GetTCPFeedback speed
    @param  [in] Default parameters flag：0-block，1-Non-blocking default1
    @return Error code success- 0,  fail-Error code
    @return Return value（The call returns successfully）[tcp_speed,ori_speed] tcp_speed Linear velocity ori_speed Position speed 
    """

    @log_call
    @xmlrpc_timeout
    def GetActualTCPCompositeSpeed(self, flag=1):
        flag = int(flag)
        # _error = self.robot.GetActualTCPCompositeSpeed(flag)
        # error = _error[0]
        # if error == 0:
        #     return error, [_error[1], _error[2]]
        # else:
        #     return error
        return 0, [self.robot_state_pkg.actual_TCP_CmpSpeed[0], self.robot_state_pkg.actual_TCP_CmpSpeed[1]]

    """   
    @brief  GetTCPCommand speed
    @param  [in] Default parameters flag：0-block，1-Non-blocking  default1
    @return Error code success- 0,  fail-Error code
    @return Return value（The call returns successfully） speed [x,y,z,rx,ry,rz]speed mm/s
    """

    @log_call
    @xmlrpc_timeout
    def GetTargetTCPSpeed(self, flag=1):
        flag = int(flag)
        # _error = self.robot.GetTargetTCPSpeed(flag)
        # error = _error[0]
        # if error == 0:
        #     return error, [_error[1], _error[2], _error[3], _error[4], _error[5], _error[6]]
        # else:
        #     return error
        return 0,[self.robot_state_pkg.target_TCP_Speed[0],self.robot_state_pkg.target_TCP_Speed[1],self.robot_state_pkg.target_TCP_Speed[2],
                  self.robot_state_pkg.target_TCP_Speed[3],self.robot_state_pkg.target_TCP_Speed[4],self.robot_state_pkg.target_TCP_Speed[5]]

    """   
    @brief  GetTCPFeedback speed
    @param  [in] Default parameters flag：0-block，1-Non-blocking  default1
    @return Error code success- 0,  fail-Error code
    @return Return value（The call returns successfully） speed [x,y,z,rx,ry,rz]speed
    """

    @log_call
    @xmlrpc_timeout
    def GetActualTCPSpeed(self, flag=1):
        flag = int(flag)
        # _error = self.robot.GetActualTCPSpeed(flag)
        # error = _error[0]
        # if error == 0:
        #     return error, [_error[1], _error[2], _error[3], _error[4], _error[5], _error[6]]
        # else:
        #     return error
        return 0,[self.robot_state_pkg.actual_TCP_Speed[0],self.robot_state_pkg.actual_TCP_Speed[1],self.robot_state_pkg.actual_TCP_Speed[2],
                  self.robot_state_pkg.actual_TCP_Speed[3],self.robot_state_pkg.actual_TCP_Speed[4],self.robot_state_pkg.actual_TCP_Speed[5]]

    """   
    @brief  Get the current tool pose
    @param  [in] Default parameters flag：0-block，1-Non-blocking default1
    @return Error code success- 0, fail-Error code
    @return Return value（The call returns successfully） tcp_pose=[x,y,z,rx,ry,rz]
    """

    @log_call
    @xmlrpc_timeout
    def GetActualTCPPose(self, flag=1):
        flag = int(flag)
        # _error = self.robot.GetActualTCPPose(flag)
        # error = _error[0]
        # if error == 0:
        #     return error, [_error[1], _error[2], _error[3], _error[4], _error[5], _error[6]]
        # else:
        #     return error
        return 0,[self.robot_state_pkg.tl_cur_pos[0],self.robot_state_pkg.tl_cur_pos[1],self.robot_state_pkg.tl_cur_pos[2],
                  self.robot_state_pkg.tl_cur_pos[3],self.robot_state_pkg.tl_cur_pos[4],self.robot_state_pkg.tl_cur_pos[5]]

    """   
    @brief  Get the current tool coordinate system number
    @param  [in] Default parameters flag：0-block，1-Non-blocking default1
    @return Error code success- 0, fail-Error code
    @return Return value（The call returns successfully） tool_id:Tool coordinate system number
    """

    @log_call
    @xmlrpc_timeout
    def GetActualTCPNum(self, flag=1):
        flag = int(flag)
        # _error = self.robot.GetActualTCPNum(flag)
        # error = _error[0]
        # if error == 0:
        #     return error, _error[1]
        # else:
        #     return error
        return 0,self.robot_state_pkg.tool

    """   
    @brief  Get the current workpiece coordinate system number 
    @param  [in] Default parameters flag：0-block，1-Non-blocking default1
    @return Error code success- 0, fail-Error code
    @return Return value（The call returns successfully） wobj_id:Workpiece coordinate system number
    """

    @log_call
    @xmlrpc_timeout
    def GetActualWObjNum(self, flag=1):
        flag = int(flag)
        # _error = self.robot.GetActualWObjNum(flag)
        # error = _error[0]
        # if error == 0:
        #     return error, _error[1]
        # else:
        #     return error
        return 0, self.robot_state_pkg.user

    """   
    @brief  Get the current end flange position
    @param  [in] Default parameters flag：0-block，1-Non-blocking default1
    @return Error code success- 0, fail-Error code
    @return Return value（The call returns successfully） flange_pose=[x,y,z,rx,ry,rz]
    """

    @log_call
    @xmlrpc_timeout
    def GetActualToolFlangePose(self, flag=1):
        flag = int(flag)
        # _error = self.robot.GetActualToolFlangePose(flag)
        # error = _error[0]
        # if error == 0:
        #     return error, [_error[1], _error[2], _error[3], _error[4], _error[5], _error[6]]
        # else:
        #     return error
        return 0,[self.robot_state_pkg.flange_cur_pos[0],self.robot_state_pkg.flange_cur_pos[1],self.robot_state_pkg.flange_cur_pos[2],
                  self.robot_state_pkg.flange_cur_pos[3],self.robot_state_pkg.flange_cur_pos[4],self.robot_state_pkg.flange_cur_pos[5]]
    """   
    @brief  Reverse Kinematics，Cartesian pose solves joint position
    @param  [in] Required parameters type:0-Absolute position (Base coordinate system)，1-Relative poses（Base coordinate system），2-Relative poses（Tool coordinate system）
    @param  [in] Required parameters desc_pose:[x,y,z,rx,ry,rz], Tool position，unit [mm][°]
    @param  [in] Default parameters config: Joint configuration，[-1]-Refer to the current joint position to solve，[0~7]-Solve according to joint configuration default-1
    @return Error code success- 0, fail-Error code
    @return Return value（The call returns successfully） joint_pos=[j1,j2,j3,j4,j5,j6]
    """

    @log_call
    @xmlrpc_timeout
    def GetInverseKin(self, type, desc_pos, config=-1):
        type = int(type)
        desc_pos = list(map(float, desc_pos))
        config = int(config)
        _error = self.robot.GetInverseKin(type, desc_pos, config)
        error = _error[0]
        if error == 0:
            return error, [_error[1], _error[2], _error[3], _error[4], _error[5], _error[6]]
        else:
            return error

    """   
    @brief  Reverse Kinematics，Tool position solution joint position，Refer to the specified joint position to solve
    @param  [in] Required parameters type:0-Absolute position (Base coordinate system)，1-Relative poses（Base coordinate system），2-Relative poses（Tool coordinate system）
    @param  [in] Required parameters desc_pose:[x,y,z,rx,ry,rz], Tool position，unit [mm][°]
    @param  [in] Required parameters joint_pos_ref：[j1,j2,j3,j4,j5,j6]，Joint reference position，unit [°]
    @return Error code success- 0,joint_pos=[j1,j2,j3,j4,j5,j6] fail-Error code
    @return Return value（The call returns successfully） joint_pos=[j1,j2,j3,j4,j5,j6]
    """

    @log_call
    @xmlrpc_timeout
    def GetInverseKinRef(self, type, desc_pos, joint_pos_ref):
        type = int(type)
        desc_pos = list(map(float, desc_pos))
        joint_pos_ref = list(map(float, joint_pos_ref))
        _error = self.robot.GetInverseKinRef(type, desc_pos, joint_pos_ref)
        error = _error[0]
        if error == 0:
            return error, [_error[1], _error[2], _error[3], _error[4], _error[5], _error[6]]
        else:
            return error

    """   
    @brief  Reverse Kinematics，Tool position to solve the joint position
    @param  [in] Required parameters type:0-Absolute position (Base coordinate system)，1-Relative poses（Base coordinate system），2-Relative poses（Tool coordinate system）
    @param  [in] Required parameters desc_pose:[x,y,z,rx,ry,rz], Tool position，unit [mm][°]
    @param  [in] Required parameters joint_pos_ref：[j1,j2,j3,j4,j5,j6]，Joint reference position，unit [°]
    @return Error code success- 0, fail-Error code
    @return Return value（The call returns successfully） result:“True”-There is a solution，“False”-No solution
    """

    @log_call
    @xmlrpc_timeout
    def GetInverseKinHasSolution(self, type, desc_pos, joint_pos_ref):
        type = int(type)
        desc_pos = list(map(float, desc_pos))
        joint_pos_ref = list(map(float, joint_pos_ref))
        _error = self.robot.GetInverseKinHasSolution(type, desc_pos, joint_pos_ref)
        error = _error[0]
        if error == 0:
            return error, _error[1]
        else:
            return error

    """   
    @brief  Orthodontics，Joint position solution tool position
    @param  [in] Required parameters joint_pos:[j1,j2,j3,j4,j5,j6]: Joint position，unit [°]
    @return Error code success- 0,  fail-Error code
    @return Return value（The call returns successfully） desc_pos=[x,y,z,rx,ry,rz]
    """

    @log_call
    @xmlrpc_timeout
    def GetForwardKin(self, joint_pos):
        joint_pos = list(map(float, joint_pos))
        _error = self.robot.GetForwardKin(joint_pos)
        error = _error[0]
        if error == 0:
            return error, [_error[1], _error[2], _error[3], _error[4], _error[5], _error[6]]
        else:
            return error

    """   
    @brief  Get the current joint torque
    @param  [in] Default parameters flag：0-block，1-Non-blocking default1
    @return Error code success- 0, fail-Error code
    @return Return value（The call returns successfully） torques=[j1,j2,j3,j4,j5,j6]
    """

    @log_call
    @xmlrpc_timeout
    def GetJointTorques(self, flag=1):
        flag = int(flag)
        # _error = self.robot.GetJointTorques(flag)
        # error = _error[0]
        # if error == 0:
        #     return error, [_error[1], _error[2], _error[3], _error[4], _error[5], _error[6]]
        # else:
        #     return error
        return 0,[self.robot_state_pkg.jt_cur_tor[0],self.robot_state_pkg.jt_cur_tor[1],self.robot_state_pkg.jt_cur_tor[2],
                  self.robot_state_pkg.jt_cur_tor[3],self.robot_state_pkg.jt_cur_tor[4],self.robot_state_pkg.jt_cur_tor[5]]

    """   
    @brief  Get the quality of the current load
    @param  [in] Default parameters flag：0-block，1-Non-blocking default1
    @return Error code success- 0, fail-Error code
    @return Return value（The call returns successfully）weight  unit [kg]
    """

    @log_call
    @xmlrpc_timeout
    def GetTargetPayload(self, flag=1):
        flag = int(flag)
        _error = self.robot.GetTargetPayload(flag)
        error = _error[0]
        if error == 0:
            return error, _error[1]
        else:
            return error

    """   
    @brief  Get the centroid of the current load
    @param  [in] Default parameters flag：0-block，1-Non-blocking default1
    @return Error code success- 0, fail-Error code
    @return Return value（The call returns successfully）cog=[x,y,z]: Centroid coordinates，unit [mm]
    """

    @log_call
    @xmlrpc_timeout
    def GetTargetPayloadCog(self, flag=1):
        flag = int(flag)
        _error = self.robot.GetTargetPayloadCog(flag)
        error = _error[0]
        if error == 0:
            return error, [_error[1], _error[2], _error[3]]
        else:
            return error

    """   
    @brief  Get the current tool coordinate system
    @param  [in] Default parameters flag：0-block，1-Non-blocking default1
    @return Error code success- 0, fail-Error code
    @return Return value（The call returns successfully）tcp_offset=[x,y,z,rx,ry,rz]: Relative poses，unit [mm][°]
    """

    @log_call
    @xmlrpc_timeout
    def GetTCPOffset(self, flag=1):
        flag = int(flag)
        _error = self.robot.GetTCPOffset(flag)
        error = _error[0]
        if error == 0:
            return error, [_error[1], _error[2], _error[3], _error[4], _error[5], _error[6]]
        else:
            return error

    """   
    @brief  Get the current workpiece coordinate system
    @param  [in] Default parameters flag：0-block，1-Non-blocking default1
    @return Error code success- 0, fail-Error code
    @return Return value（The call returns successfully）wobj_offset=[x,y,z,rx,ry,rz]: Relative poses，unit [mm][°]
    """

    @log_call
    @xmlrpc_timeout
    def GetWObjOffset(self, flag=1):
        flag = int(flag)
        _error = self.robot.GetWObjOffset(flag)
        error = _error[0]
        if error == 0:
            return error, [_error[1], _error[2], _error[3], _error[4], _error[5], _error[6]]
        else:
            return error

    """   
    @brief  Get the soft limit angle of joints
    @param  [in] Default parameters flag：0-block，1-Non-blocking default1
    @return Error code success- 0, fail-Error code
    @return Return value（The call returns successfully）[j1min,j1max,j2min,j2max,j3min,j3max,j4min,j4max,j5min,j5max,j6min,j6max]: axis 1~ axis 6 Negative and positive limits of joints，unit [mm]
    """

    @log_call
    @xmlrpc_timeout
    def GetJointSoftLimitDeg(self, flag=1):
        flag = int(flag)
        _error = self.robot.GetJointSoftLimitDeg(flag)
        error = _error[0]
        if error == 0:
            return error, [_error[1], _error[2], _error[3], _error[4], _error[5], _error[6], _error[7], _error[8],
                           _error[9], _error[10], _error[11], _error[12]]
        else:
            return error

    """   
    @brief  Get system time
    @param  [in] NULL
    @return Error code success- 0, fail-Error code
    @return Return value（The call returns successfully）t_ms: unit [ms]
    """

    @log_call
    @xmlrpc_timeout
    def GetSystemClock(self):
        _error = self.robot.GetSystemClock()
        error = _error[0]
        if error == 0:
            return error, _error[1]
        else:
            return error

    """   
    @brief  Get the current joint configuration of the robot
    @param  [in] NULL
    @return Error code success- 0, fail-Error code
    @return Return value（The call returns successfully）config: scope [0~7]
    """

    @log_call
    @xmlrpc_timeout
    def GetRobotCurJointsConfig(self):
        _error = self.robot.GetRobotCurJointsConfig()
        error = _error[0]
        if error == 0:
            return error, _error[1]
        else:
            return error

    """   
    @brief  Get the default speed
    @param  [in] NULL
    @return Error code success- 0, fail-Error code
    @return Return value（The call returns successfully）vel: unit [mm/s]
    """

    @log_call
    @xmlrpc_timeout
    def GetDefaultTransVel(self):
        _error = self.robot.GetDefaultTransVel()
        error = _error[0]
        if error == 0:
            return error, _error[1]
        else:
            return error

    """   
    @brief  Check whether the robot's movement is completed
    @param  [in] NULL
    @return Error code success- 0, fail-Error code
    @return Return value（The call returns successfully）state:0-Not completed，1-Finish
    """

    @log_call
    @xmlrpc_timeout
    def GetRobotMotionDone(self):
        # _error = self.robot.GetRobotMotionDone()
        # error = _error[0]
        # if error == 0:
        #     return error, _error[1]
        # else:
        #     return error
            return 0,self.robot_state_pkg.motion_done
    """   
    @brief  Query the robot error code
    @param  [in] NULL
    @return Error code success- 0, fail-Error code
    @return Return value（The call returns successfully）[maincode subcode] maincode Main error code subcode Sub-Error Code
    """

    @log_call
    @xmlrpc_timeout
    def GetRobotErrorCode(self):
        # _error = self.robot.GetRobotErrorCode()
        # error = _error[0]
        # if error == 0:
        #     return error, [_error[1], _error[2]]
        # else:
        #     return error
        return 0, [self.robot_state_pkg.main_code,self.robot_state_pkg.sub_code]

    """   
    @brief  Query robot teaching management point data
    @param  [in] Required parameters name  Point name
    @return Error code success- 0, fail-Error code
    @return Return value（The call returns successfully）data Point data[x,y,z,rx,ry,rz,j1,j2,j3,j4,j5,j6,tool, wobj,speed,acc,e1,e2,e3,e4]
    """

    @log_call
    @xmlrpc_timeout
    def GetRobotTeachingPoint(self, name):
        name = str(name)
        _error = self.robot.GetRobotTeachingPoint(name)
        error = _error[0]
        if error == 0:
            data =_error[1].split(',')
            if len(data)!= 20:
                self.log_error("get get Teaching Point size fail")
                return -1
            return error, [data[0],data[1], data[2], data[3], data[4], data[5], data[6],data[7],
                           data[8], data[9], data[10], data[11], data[12], data[13], data[14], data[15],
                           data[16], data[17], data[18], data[19] ]
        else:
            return error

    """   
    @brief  Query the cache length of the robot motion queue
    @param  [in] NULL
    @return Error code success- 0, fail-Error code
    @return Return value（The call returns successfully）len  Cache length
    """

    @log_call
    @xmlrpc_timeout
    def GetMotionQueueLength(self):
        # _error = self.robot.GetMotionQueueLength()
        # error = _error[0]
        # if error == 0:
        #     return error, _error[1]
        # else:
        #     return error
        return 0, self.robot_state_pkg.mc_queue_len

    """   
    @brief  Get the robot emergency stop state
    @param  [in] NULL
    @return Error code success- 0, fail-Error code
    @return Return value（The call returns successfully）state Emergency stop state，0-No emergency stop，1-Emergency stop
    """

    @log_call
    @xmlrpc_timeout
    def GetRobotEmergencyStopState(self):
        # _error = self.robot.GetRobotEmergencyStopState()
        # error = _error[0]
        # if error == 0:
        #     return error, _error[1]
        # else:
        #     return error
        return 0, self.robot_state_pkg.EmergencyStop

    """   
    @brief  Get a safe stop signal
    @param  [in] NULL
    @return Error code success- 0, fail-Error code
    @return Return value（The call returns successfully）[si0_state,si1_state] si0_state Safe stop signalSI0，0-invalid，1-efficient si1_state Safe stop signalSI1，0-invalid，1-efficient
    """

    @log_call
    @xmlrpc_timeout
    def GetSafetyStopState(self):
        # _error = self.robot.GetSafetyStopState()
        # error = _error[0]
        # if error == 0:
        #     return error, [_error[1], _error[2]]
        # else:
        #     return error

        return 0, [self.robot_state_pkg.safety_stop0_state,self.robot_state_pkg.safety_stop1_state]

    """   
    @brief  GetSDKCommunication status with robot
    @param  [in] NULL
    @return Error code success- 0, fail-Error code
    @return Return value（The call returns successfully）state Communication status，0-Communication is normal，1-Communication abnormality
    """

    @log_call
    @xmlrpc_timeout
    def GetSDKComState(self):
        _error = self.robot.GetSDKComState()
        error = _error[0]
        if error == 0:
            return error, [_error[1], _error[2]]
        else:
            return error

    """   
       @brief  GetSSHPublic Key
       @param  [in] NULL
       @return Error code success-0，fail-Error code
       @return Return value（The call returns successfully） keygen Public Key
       """

    @log_call
    @xmlrpc_timeout
    def GetSSHKeygen(self):
        _error = self.robot.GetSSHKeygen()
        error = _error[0]
        if _error[0] == 0:
            return error, _error[1]
        else:
            return error

    """   
    @brief  PostedSCPinstruction
    @param  [in] Required parameters mode 0-Upload（Upload machine->Controller），1-download（Controller->Upload machine）
    @param  [in] Required parameters sshname Upper computer username
    @param  [in] Required parameters sship Upload machineipaddress
    @param  [in] Required parameters usr_file_url File path of the host computer
    @param  [in] Required parameters robot_file_url Robot controller file path
    @return Error code success-0  fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def SetSSHScpCmd(self, mode, sshname, sship, usr_file_url, robot_file_url):
        mode = int(mode)
        sshname = str(sshname)
        sship = str(sship)
        usr_file_url = str(usr_file_url)
        robot_file_url = str(robot_file_url)
        error = self.robot.SetSSHScpCmd(mode, sshname, sship, usr_file_url, robot_file_url)
        return error

    """   
    @brief  Calculate the file under the specified pathMD5value
    @param  [in] Required parameters file_path File path contains file name，defaultTrajThe folder path is:"/fruser/traj/",like"/fruser/traj/trajHelix_aima_1.txt"
    @return Error code success-0  fail-Error code
    @return Return value（The call returns successfully）md5 documentMD5value
    """

    @log_call
    @xmlrpc_timeout
    def ComputeFileMD5(self, file_path):
        file_path = str(file_path)
        _error = self.robot.ComputeFileMD5(file_path)
        error = _error[0]
        if _error[0] == 0:
            return error, _error[1]
        else:
            return error

    """   
    @brief  Get robot version information
    @param  [in] NULL
    @return Error code success-0  fail-Error code
    @return Return value（The call returns successfully） robotModel Robot model
    @return Return value（The call returns successfully） webVersion webVersion
    @return Return value（The call returns successfully） controllerVersion Controller version
    """

    @log_call
    @xmlrpc_timeout
    def GetSoftwareVersion(self):
        _error = self.robot.GetSoftwareVersion()
        error = _error[0]
        if _error[0] == 0:
            return error, _error[1], _error[2], _error[3]
        else:
            return error

    """   
    @brief  Get robot hardware version information
    @param  [in] NULL
    @return Error code success-0  fail-Error code
    @return Return value（The call returns successfully） ctrlBoxBoardVersion Control box version
    @return Return value（The call returns successfully） driver1Version 
    @return Return value（The call returns successfully） driver2Version 
    @return Return value（The call returns successfully） driver3Version
    @return Return value（The call returns successfully） driver4Version
    @return Return value（The call returns successfully） driver5Version
    @return Return value（The call returns successfully） driver6Version
    @return Return value（The call returns successfully） endBoardVersion
    """

    @log_call
    @xmlrpc_timeout
    def GetSlaveHardVersion(self):
        _error = self.robot.GetSlaveHardVersion()
        error = _error[0]
        if _error[0] == 0:
            return error, _error[1], _error[2], _error[3], _error[4], _error[5], _error[6], _error[7], _error[8]
        else:
            return error

    """   
    @brief  Get robot firmware version information
    @param  [in] NULL
    @return Error code success-0  fail-Error code
    @return Return value（The call returns successfully） ctrlBoxBoardVersion Control box version
    @return Return value（The call returns successfully） driver1Version 
    @return Return value（The call returns successfully） driver2Version 
    @return Return value（The call returns successfully） driver3Version
    @return Return value（The call returns successfully） driver4Version
    @return Return value（The call returns successfully） driver5Version
    @return Return value（The call returns successfully） driver6Version
    @return Return value（The call returns successfully） endBoardVersion
    """

    @log_call
    @xmlrpc_timeout
    def GetSlaveFirmVersion(self):
        _error = self.robot.GetSlaveFirmVersion()
        error = _error[0]
        if _error[0] == 0:
            return error, _error[1], _error[2], _error[3], _error[4], _error[5], _error[6], _error[7], _error[8]
        else:
            return error

    """   
    @brief  GetDHCompensation parameters
    @param  [in] NULL
    @return Error code success-0  fail-Error code
    @return Return value（The call returns successfully） dhCompensation robotDHParameter compensation value(mm) [cmpstD1,cmpstA2,cmpstA3,cmpstD4,cmpstD5,cmpstD6]
    """

    @log_call
    @xmlrpc_timeout
    def GetDHCompensation(self):
        _error = self.robot.GetDHCompensation()
        error = _error[0]
        if _error[0] == 0:
            return error, [_error[1], _error[2], _error[3], _error[4], _error[5], _error[6]]
        else:
            return error

    """   
    ***************************************************************************Robot trajectory reappearance********************************************************************************************
    """

    """   
    @brief  Set track record parameters
    @param  [in] Required parameters name：Trajectory name
    @param  [in] Required parameters period_ms：Sampling cycle，Fixed value，2ms or 4ms or 8ms
    @param  [in] Default parameters type：Data Type，1-Joint position default1
    @param  [in] Default parameters di_choose：DI choose,bit0~bit7 Corresponding control box DI0~DI7，bit8~bit9 Corresponding endDI0~DI1，0-Not selected，1-choose default0
    @param  [in] Default parameters do_choose：DO choose,bit0~bit7 Corresponding control box DO0~DO7，bit8~bit9 Corresponding end DO0~DO1，0-Not selected，1-choose default0
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def SetTPDParam(self, name, period_ms, type=1, di_choose=0, do_choose=0):
        name = str(name)
        period_ms = int(period_ms)
        type = int(type)
        di_choose = int(di_choose)
        do_choose = int(do_choose)
        error = self.robot.SetTPDParam(type, name, period_ms, di_choose, do_choose)
        return error

    """   
    @brief  Start track record
    @param  [in] Required parameters name：Trajectory name
    @param  [in] Required parameters period_ms：Sampling cycle，Fixed value，2ms or 4ms or 8ms
    @param  [in] Default parameters type：Data Type，1-Joint position default1
    @param  [in] Default parameters di_choose：DI choose,bit0~bit7 Corresponding control box DI0~DI7，bit8~bit9 Corresponding endDI0~DI1，0-Not selected，1-choose default0
    @param  [in] Default parameters do_choose：DO choose,bit0~bit7 Corresponding control box DO0~DO7，bit8~bit9 Corresponding end DO0~DO1，0-Not selected，1-choose default0
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def SetTPDStart(self, name, period_ms, type=1, di_choose=0, do_choose=0):
        name = str(name)
        period_ms = int(period_ms)
        type = int(type)
        di_choose = int(di_choose)
        do_choose = int(do_choose)
        error = self.robot.SetTPDStart(type, name, period_ms, di_choose, do_choose)
        return error

    """   
    @brief  Stop track record
    @param  [in] NULL
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def SetWebTPDStop(self):
        error = self.robot.SetWebTPDStop()
        return error

    """   
    @brief  Delete track record
    @param  [in] Required parameters name：Trajectory name
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def SetTPDDelete(self, name):
        name = str(name)
        error = self.robot.SetTPDDelete(name)
        return error

    """   
    @brief  Track preload
    @param  [in] Required parameters name：Trajectory name
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def LoadTPD(self, name):
        name = str(name)
        error = self.robot.LoadTPD(name)
        return error

    """   
    @brief  Get the starting position of the track
    @param  [in] name Track file name,No file suffix required
    @return Error code success- 0, fail-Error code
    @return Return value（The call returns successfully）desc_pose [x,y,z,rx,ry,rz]
    """

    @log_call
    @xmlrpc_timeout
    def GetTPDStartPose(self, name):
        name = str(name)
        _error = self.robot.GetTPDStartPose(name)
        error = _error[0]
        if error == 0:
            return error, [_error[1], _error[2], _error[3], _error[4], _error[5], _error[6]]
        else:
            return error

    """   
    @brief  Trajectory reappearance
    @param  [in] Required parameters name：Trajectory name
    @param  [in] Required parameters blend：Is it smooth or not，0-Non-smooth，1-smooth
    @param  [in] Required parameters ovl：Speed ​​scaling factor，scope [0~100]
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def MoveTPD(self, name, blend, ovl):
        if self.GetSafetyCode() != 0:
            return self.GetSafetyCode()
        name = str(name)
        blend = int(blend)
        ovl = float(ovl)
        error = self.robot.MoveTPD(name, blend, ovl)
        return error

    """   
    @brief  Trajectory preprocessing
    @param  [in] Required parameters name：Trajectory name like/fruser/traj/trajHelix_aima_1.txt
    @param  [in] Required parameters ovl Speed ​​scaling percentage，scope[0~100]
    @param  [in] Default parameters opt 1-Control Point，Default is1
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def LoadTrajectoryJ(self, name, ovl, opt=1):
        name = str(name)
        ovl = float(ovl)
        opt = int(opt)
        error = self.robot.LoadTrajectoryJ(name, ovl, opt)
        return error

    """   
    @brief  Trajectory reappearance
    @param  [in] NULL
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def MoveTrajectoryJ(self):
        if self.GetSafetyCode() != 0:
            return self.GetSafetyCode()
        error = self.robot.MoveTrajectoryJ()
        return error

    """   
    @brief  Get the starting position of the track
    @param  [in] Required parameters name：Trajectory name
    @return Error code success- 0, fail-Error code
    @return Return value（The call returns successfully）desc_pose [x,y,z,rx,ry,rz]
    """

    @log_call
    @xmlrpc_timeout
    def GetTrajectoryStartPose(self, name):
        name = str(name)
        _error = self.robot.GetTrajectoryStartPose(name)
        error = _error[0]
        if error == 0:
            return error, [_error[1], _error[2], _error[3], _error[4], _error[5], _error[6]]
        else:
            return error

    """   
    @brief  Get track point number
    @param  [in] NULL
    @return Error code success- 0, fail-Error code
    @return Return value（The call returns successfully）pnum
    """

    @log_call
    @xmlrpc_timeout
    def GetTrajectoryPointNum(self):
        _error = self.robot.GetTrajectoryPointNum()
        error = _error[0]
        if error == 0:
            return error, _error[1]
        else:
            return error

    """   
    @brief  Set the speed of the track running
    @param  [in] Required parameters ovl Speed ​​scaling percentage，scope[0~100]
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def SetTrajectoryJSpeed(self, ovl):
        ovl = float(ovl)
        error = self.robot.SetTrajectoryJSpeed(ovl)
        return error

    """   
    @brief  Set the force and torque during trajectory operation
    @param  [in] Required parameters ft [fx,fy,fz,tx,ty,tz]，unitNandNm
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def SetTrajectoryJForceTorque(self, ft):
        ft = list(map(float, ft))
        error = self.robot.SetTrajectoryJForceTorque(ft)
        return error

    """   
    @brief  Set the edge of the trackxDirectional force
    @param  [in] Required parameters fx alongxDirectional force，unitN
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def SetTrajectoryJForceFx(self, fx):
        fx = float(fx)
        error = self.robot.SetTrajectoryJForceFx(fx)
        return error

    """   
    @brief  Set the edge of the trackyDirectional force
    @param  [in] Required parameters fy alongyDirectional force，unitN
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def SetTrajectoryJForceFy(self, fy):
        fy = float(fy)
        error = self.robot.SetTrajectoryJForceFy(fy)
        return error

    """   
    @brief  Set the edge of the trackzDirectional force
    @param  [in] Required parameters fz alongzDirectional force，unitN
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def SetTrajectoryJForceFz(self, fz):
        fz = float(fz)
        error = self.robot.SetTrajectoryJForceFy(fz)
        return error

    """   
    @brief  Set the trajectory running swayxThe torque of the shaft
    @param  [in] Required parameters tx CirclexThe torque of the shaft，unitNm
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def SetTrajectoryJTorqueTx(self, tx):
        tx = float(tx)
        error = self.robot.SetTrajectoryJTorqueTx(tx)
        return error

    """   
    @brief  Set the trajectory running swayyThe torque of the shaft
    @param  [in] Required parameters ty CircleyThe torque of the shaft，unitNm
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def SetTrajectoryJTorqueTy(self, ty):
        ty = float(ty)
        error = self.robot.SetTrajectoryJTorqueTx(ty)
        return error

    """   
    @brief  Set the trajectory running swayzThe torque of the shaft
    @param  [in] Required parameters tz CirclezThe torque of the shaft，unitNm
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def SetTrajectoryJTorqueTz(self, tz):
        tz = float(tz)
        error = self.robot.SetTrajectoryJTorqueTx(tz)
        return error

    """   
    ***************************************************************************robotWebAPPProgram usage********************************************************************************************
    """

    """   
    @brief  Set the default job program to automatically load on the computer
    @param  [in] Required parameters flag：0-The default program does not automatically load on the computer，1-Automatically load the default program on the startup
    @param  [in] Required parameters program_name：Job program name and path，like“/fruser/movej.lua”，in“/fruser/”For fixed path
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def LoadDefaultProgConfig(self, flag, program_name):
        flag = int(flag)
        program_name = str(program_name)
        error = self.robot.LoadDefaultProgConfig(flag, program_name)
        return error

    """   
    @brief  Load the specified job program
    @param  [in] Required parameters program_name：Job program name and path，like“/fruser/movej.lua”，in“/fruser/”For fixed path
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def ProgramLoad(self, program_name):
        program_name = str(program_name)
        error = self.robot.ProgramLoad(program_name)
        return error

    """   
    @brief  Get the execution line number of the current robot job program
    @param  [in] NULL
    @return Error code success- 0, fail-Error code
    @return Return value（The call returns successfully）line_num
    """

    @log_call
    @xmlrpc_timeout
    def GetCurrentLine(self):
        _error = self.robot.GetCurrentLine()
        error = _error[0]
        if error == 0:
            return error, _error[1]
        else:
            return error

    """   
    @brief  Run the currently loaded job program
    @param  [in] NULL
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def ProgramRun(self):
        if self.GetSafetyCode() != 0:
            return self.GetSafetyCode()
        error = self.robot.ProgramRun()
        return error

    """   
    @brief  Pause the currently running job program
    @param  [in] NULL
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def ProgramPause(self):
        error = self.robot.ProgramPause()
        return error

    """   
    @brief  Resuming the currently suspended job program
    @param  [in] NULL
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def ProgramResume(self):
        if self.GetSafetyCode() != 0:
            return self.GetSafetyCode()
        error = self.robot.ProgramResume()
        return error

    """   
    @brief  Terminate the currently running job program
    @param  [in] NULL
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def ProgramStop(self):
        error = self.robot.ProgramStop()
        return error

    """   
    @brief  Get the execution status of the robot job program
    @param  [in] NULL
    @return Error code success- 0, fail-Error code
    @return Return value（The call returns successfully）state:1-Program stops or no program runs，2-The program is running，3-Program pause
    """

    @log_call
    @xmlrpc_timeout
    def GetProgramState(self):
        # _error = self.robot.GetProgramState()
        # error = _error[0]
        # if error == 0:
        #     return error, _error[1]
        # else:
        #     return error
        return 0,self.robot_state_pkg.robot_state

    """   
    @brief  Get the loaded job program name
    @param  [in] NULL
    @return Error code success- 0, fail-Error code
    @return Return value（The call returns successfully）program_name
    """

    @log_call
    @xmlrpc_timeout
    def GetLoadedProgram(self):
        _error = self.robot.GetLoadedProgram()
        error = _error[0]
        if error == 0:
            return error, _error[1]
        else:
            return error

    """   
    ***************************************************************************Robot peripherals********************************************************************************************
    """

    """   
    @brief  Get the jaw configuration
    @param  [in] NULL
    @return Error code success- 0, fail-Error code
    @return Return value（The call returns successfully）[number,company,device,softversion] 
            number Claw number
            companyClaw manufacturer，1-Robotiq，2-Huiling，3-Heavenly news，4-Dahuan，5-Knowledge and Action 
            device  Equipment number，Robotiq(0-2F-85series)，Huiling(0-NKseries,1-Z-EFG-100)，Heavenly news(0-TEG-110)，Dahuan(0-PGI-140)，Knowledge and Action(0-CTPM2F20)
            softvesion  Software version number，Not used yet，Default is0
    """

    @log_call
    @xmlrpc_timeout
    def GetGripperConfig(self):
        _error = self.robot.GetGripperConfig()
        error = _error[0]
        if error == 0:
            return error, [_error[1] + 1, _error[2] + 1, _error[3], _error[4]]
        else:
            return error

    """   
    @brief  Activate the claw
    @param  [in] Required parameters index: Claw number
    @param  [in] Required parameters action:0-Reset，1-activation
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def ActGripper(self, index, action):
        index = int(index)
        action = int(action)
        error = self.robot.ActGripper(index, action)
        return error

    """   
    @brief  Control claws
    @param  [in] Required parameters index: Claw number
    @param  [in] Required parameters pos: Position percentage，scope [0~100]
    @param  [in] Required parameters vel: Speed ​​percentage，scope [0~100]
    @param  [in] Required parameters force: Torque Percentage，scope [0~100]
    @param  [in] Required parameters maxtime: Maximum waiting time，scope [0~30000]，unit [ms]
    @param  [in] Required parameters block:0-block，1-Non-blocking
    @param  [in] Required parameters type Claw type，0-Parallel claws；1-Rotating claws
    @param  [in] Required parameters rotNum Number of rotations
    @param  [in] Required parameters rotVel Percent rotation speed[0-100]
    @param  [in] Required parameters rotTorque Percent rotation torque[0-100]
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def MoveGripper(self, index, pos, vel, force, maxtime, block, type, rotNum, rotVel, rotTorque):
        if self.GetSafetyCode() != 0:
            return self.GetSafetyCode()
        index = int(index)
        pos = int(pos)
        vel = int(vel)
        force = int(force)
        maxtime = int(maxtime)
        block = int(block)
        type = int(type)
        rotNum = float(rotNum)
        rotVel = int(rotVel)
        rotTorque = int(rotTorque)
        error = self.robot.MoveGripper(index, pos, vel, force, maxtime, block, type, rotNum, rotVel, rotTorque)
        return error

    """   
    @brief  Obtain the jaw movement status
    @param  [in] NULL
    @return Error code success- 0, fail-Error code
    @return Return value（The call returns successfully）[fault,status]：Claw movement status，fault:0-No errors，1-There is an error；status:0-The movement is not completed，1-Exercise completion    
    """

    @log_call
    @xmlrpc_timeout
    def GetGripperMotionDone(self):
        _error = self.robot.GetGripperMotionDone()
        error = _error[0]
        if error == 0:
            return error, [_error[1], _error[2]]
        else:
            return error

    """   
    @brief  Configure jaws
    @param  [in] Required parameters company：Claw manufacturer，1-Robotiq，2-Huiling，3-Heavenly news，4-Dahuan，5-Knowledge and Action
    @param  [in] Required parameters device：Equipment number，Robotiq(0-2F-85 series)，Huiling (0-NK series,1-Z-EFG-100)，Heavenly news (0-TEG-110)，Dahuan (0-PGI-140)，Knowledge and Action (0-CTPM2F20)
    @param  [in] Default parameters softversion：Software version number，Not used yet，Default is 0
    @param  [in] Default parameters bus：Device mounting end bus position，Not used yet，Default is 0；
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def SetGripperConfig(self, company, device, softversion=0, bus=0):
        company = int(company)
        device = int(device)
        softversion = int(softversion)
        bus = int(bus)
        error = self.robot.SetGripperConfig(company, device, softversion, bus)
        return error

    """   
    @brief  Calculate precrawl points-Vision
    @param  [in] Required parameters desc_pos  Catch Point Cartesian Position
    @param  [in] Required parameters zlength   zAxis offset
    @param  [in] Required parameters zangle    CirclezAxis rotation offset
    @return Error code success- 0, fail-Error code
    @return Return value（The call returns successfully） pre_pos  Pre-catch point Cartesian pose
    """

    @log_call
    @xmlrpc_timeout
    def ComputePrePick(self, desc_pos, zlength, zangle):
        desc_pos = list(map(float, desc_pos))
        zlength = float(zlength)
        zangle = float(zangle)
        _error = self.robot.ComputePrePick(desc_pos, zlength, zangle)
        error = _error[0]
        if error == 0:
            return error, [_error[1], _error[2], _error[3], _error[4], _error[5], _error[6]]
        else:
            return error

    """   
    @brief  Calculate retreat points-Vision
    @param  [in] Required parameters desc_pos  Catch Point Cartesian Position
    @param  [in] Required parameters zlength   zAxis offset
    @param  [in] Required parameters zangle    CirclezAxis rotation offset
    @return Error code success- 0, fail-Error code
    @return Return value（The call returns successfully） post_pos  Retreat Point Cartesian Position
    """

    @log_call
    @xmlrpc_timeout
    def ComputePostPick(self, desc_pos, zlength, zangle):
        desc_pos = list(map(float, desc_pos))
        zlength = float(zlength)
        zangle = float(zangle)
        _error = self.robot.ComputePostPick(desc_pos, zlength, zangle)
        error = _error[0]
        if error == 0:
            return error, [_error[1], _error[2], _error[3], _error[4], _error[5], _error[6]]
        else:
            return error

    """   
    ***************************************************************************Robot power control********************************************************************************************
    """

    """   
    @brief  Obtain force sensor configuration
    @param  [in] NULL
    @return Error code success- 0, fail-Error code
    @return Return value（The call returns successfully）[number,company,device,softversion,bus]
            number Sensor number
            company  Force sensor manufacturer，17-Kunwei Technology，19-The Eleventh Academy of Space，20-ATI sensor，21-Zhongcomi Point，22-Weihang Analytical Core
            device  Equipment number，Kunwei (0-KWR75B)，The Eleventh Academy of Space (0-MCS6A-200-4)，ATI(0-AXIA80-M8)，Zhongcomi Point (0-MST2010)，Weihang Analytical Core (0-WHC6L-YB10A)
            softvesion  Software version number，Not used yet，Default is0    
    """

    @log_call
    @xmlrpc_timeout
    def FT_GetConfig(self):
        _error = self.robot.FT_GetConfig()
        error = _error[0]
        if error == 0:
            return error, [_error[1] + 1, _error[2] + 1, _error[3], _error[4]]
        else:
            return error

    """   
    @brief  Force sensor configuration
    @param  [in] Required parameters company：Sensor manufacturer，17-Kunwei Technology，19-The Eleventh Academy of Space，20-ATI sensor，21-Zhongcomi Point，22-Weihang Analytical Core，23-NBIT，24-Xin Jingcheng(XJC)，26-NSR；
    @param  [in] Required parameters device：Equipment number，Kunwei (0-KWR75B)，The Eleventh Academy of Space (0-MCS6A-200-4)，ATI(0-AXIA80-M8)，Zhongcomi Point (0-MST2010)，Weihang Analytical Core (0-WHC6L-YB10A)，NBIT(0-XLH93003ACS)，Xin JingchengXJC(0-XJC-6F-D82)，NSR(0-NSR-FTSensorA)；
    @param  [in] Default parameters softversion：Software version number，Not used yet，Default is 0
    @param  [in] Default parameters bus：Device mounting end bus position，Not used yet，Default is 0；
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def FT_SetConfig(self, company, device, softversion=0, bus=0):
        company = int(company)
        device = int(device)
        softversion = int(softversion)
        bus = int(bus)
        error = self.robot.FT_SetConfig(company, device, softversion, bus)
        return error

    """   
    @brief  Force sensor activation
    @param  [in] Required parameters state：0-Reset，1-activation
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def FT_Activate(self, state):
        state = int(state)
        error = self.robot.FT_Activate(state)
        return error

    """   
    @brief  Force sensor zero correction
    @param  [in] Required parameters state：0-Remove zero points，1-Zero correction
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def FT_SetZero(self, state):
        state = int(state)
        error = self.robot.FT_SetZero(state)
        return error

    """   
    @brief  Set the force sensor reference coordinate system
    @param  [in] Required parameters ref：0-Tool coordinate system，1-Base coordinate system
    @param  [in] Default parameters coord：[x,y,z,rx,ry,rz] Custom coordinate system values,default[0,0,0,0,0,0]
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def FT_SetRCS(self, ref,coord=[0,0,0,0,0,0]):
        ref = int(ref)
        coord = list(map(float, coord))
        error = self.robot.FT_SetRCS(ref,coord)
        return error

    """   
    @brief  Load weight identification calculation
    @param  [in] NULL
    @return Error code success- 0, fail-Error code
    @return Return value（The call returns successfully）weight-Load weight，unit kg
    """

    @log_call
    @xmlrpc_timeout
    def FT_PdIdenCompute(self):
        _error = self.robot.FT_PdIdenCompute()
        error = _error[0]
        if error == 0:
            return error, _error[1]
        else:
            return error

    """   
    @brief  Load weight identification record
    @param  [in] Required parameters tool_id：Sensor coordinate system number，scope [1~14]
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def FT_PdIdenRecord(self, tool_id):
        tool_id = int(tool_id)
        error = self.robot.FT_PdIdenRecord(tool_id)
        return error

    """   
    @brief  Load centroid identification calculation
    @param  [in] NULL
    @return Error code success- 0, fail-Error code
    @return Return value（The call returns successfully）cog=[cogx,cogy,cogz] ，Load centroid，unit mm
    """

    @log_call
    @xmlrpc_timeout
    def FT_PdCogIdenCompute(self):
        _error = self.robot.FT_PdCogIdenCompute()
        error = _error[0]
        if error == 0:
            return error, [_error[1], _error[2], _error[3]]
        else:
            return error

    """   
    @brief  Load centroid identification record
    @param  [in] Required parameters tool_id：Sensor coordinate system number，scope [0~14]
    @param  [in] Required parameters index Point number，scope[1~3]
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def FT_PdCogIdenRecord(self, tool_id, index):
        tool_id = int(tool_id)
        index = int(index)
        error = self.robot.FT_PdCogIdenRecord(tool_id, index)
        return error

    """   
    @brief  Obtain the force of the reference coordinate system/Torque data
    @param  [in] Required parameters NULL
    @return Error code success- 0, fail-Error code
    @return Return value（The call returns successfully）data=[fx,fy,fz,tx,ty,tz]
    """

    @log_call
    @xmlrpc_timeout
    def FT_GetForceTorqueRCS(self):
        # _error = self.robot.FT_GetForceTorqueRCS(0)
        # error = _error[0]
        # if error == 0:
        #     return error, [_error[1], _error[2], _error[3], _error[4], _error[5], _error[6]]
        # else:
        #     return error
        return 0,[self.robot_state_pkg.ft_sensor_data[0],self.robot_state_pkg.ft_sensor_data[1],self.robot_state_pkg.ft_sensor_data[2],
                  self.robot_state_pkg.ft_sensor_data[3],self.robot_state_pkg.ft_sensor_data[4],self.robot_state_pkg.ft_sensor_data[5]]

    """   
    @brief  Obtain the original force of the force sensor/Torque data
    @param  [in] Required parameters NULL
    @return Error code success- 0, fail-Error code
    @return Return value（The call returns successfully）data=[fx,fy,fz,tx,ty,tz]
    """

    @log_call
    @xmlrpc_timeout
    def FT_GetForceTorqueOrigin(self):
        # _error = self.robot.FT_GetForceTorqueOrigin(0)
        # error = _error[0]
        # return error
        # if error == 0:
        #     return error, [_error[1], _error[2], _error[3], _error[4], _error[5], _error[6]]
        # else:
        #     return error
        return 0,[self.robot_state_pkg.ft_sensor_raw_data[0],self.robot_state_pkg.ft_sensor_raw_data[1],self.robot_state_pkg.ft_sensor_raw_data[2],
                  self.robot_state_pkg.ft_sensor_raw_data[3],self.robot_state_pkg.ft_sensor_raw_data[4],self.robot_state_pkg.ft_sensor_raw_data[5]]

    """   
    @brief  Collision Guard
    @param  [in] Required parameters flag：0-Turn off the collision guard，1-Turn on collision protection；
    @param  [in] Required parameters sensor_num：Force sensor number
    @param  [in] Required parameters select：Whether the six degrees of freedom detects collisions [fx,fy,fz,mx,my,mz]，0-Not effective，1-Take effect
    @param  [in] Required parameters force_torque：Collision detection force/Torque，unit N or Nm
    @param  [in] Required parameters max_threshold：Maximum threshold
    @param  [in] Required parameters min_threshold：Minimum threshold
    force/Torque detection range:(force_torque-min_threshold,force_torque+max_threshold)
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def FT_Guard(self, flag, sensor_num, select, force_torque, max_threshold, min_threshold):
        flag = int(flag)
        sensor_num = int(sensor_num)
        select = list(map(int, select))
        force_torque = list(map(float, force_torque))
        max_threshold = list(map(float, max_threshold))
        min_threshold = list(map(float, min_threshold))
        error = self.robot.FT_Guard(flag, sensor_num, select, force_torque, max_threshold, min_threshold)
        return error

    """   
    @brief  Constant Power Control
    @param  [in] Required parameters flag：0-Turn off the collision guard，1-Turn on collision protection；
    @param  [in] Required parameters sensor_num：Force sensor number
    @param  [in] Required parameters select：[fx,fy,fz,mx,my,mz]Whether the six degrees of freedom detects collisions ，0-Not effective，1-Take effect
    @param  [in] Required parameters force_torque：[fx,fy,fz,mx,my,mz]Collision detection force/Torque，unit N or Nm
    @param  [in] Required parameters gain：[f_p,f_i,f_d,m_p,m_i,m_d], forcePIDparameter，TorquePIDparameter
    @param  [in] Required parameters adj_sign：Adaptive start and stop state，0-closure，1-Open
    @param  [in] Required parameters ILC_sign: ILC Control start and stop status，0-stop，1-train，2-Practical
    @param  [in] Required parameters max_dis：Maximum adjustment distance，unitmm
    @param  [in] Required parameters max_ang：Maximum adjustment angle，unitdeg
    @param  [in] Required parameters r：Radius of sanding disc，unitmm
    @param  [in] Default parameters filter_Sign Filtering on flag 0-close；1-open，default 0-closure
    @param  [in] Default parameters posAdapt_sign The posture conforms to the opening sign 0-close；1-open，default 0-closure
    @param  [in] Default parameters isNoBlock Blocking flag，0-block；1-Non-blocking default0-block
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def FT_Control(self, flag, sensor_num, select, force_torque, gain, adj_sign, ILC_sign, max_dis, max_ang,filter_Sign=0, posAdapt_sign=0,isNoBlock=0):
        flag = int(flag)
        sensor_num = int(sensor_num)
        select = list(map(int, select))
        force_torque = list(map(float, force_torque))
        gain = list(map(float, gain))
        adj_sign = int(adj_sign)
        ILC_sign = int(ILC_sign)
        max_dis = float(max_dis)
        max_ang = float(max_ang)

        filter_Sign = int(filter_Sign)
        posAdapt_sign = int(posAdapt_sign)
        isNoBlock = int(isNoBlock)
        error = self.robot.FT_Control(flag, sensor_num, select, force_torque, gain, adj_sign, ILC_sign, max_dis,
                                      max_ang, filter_Sign, posAdapt_sign,isNoBlock)
        return error

    """   
    @brief  Spiral Exploration
    @param  [in] Required parameters rcs Reference coordinate system，0-Tool coordinate system，1-Base coordinate system
    @param  [in] Required parameters ft：Force or torque threshold (0~100)，unit N or Nm
    @param  [in] Default parameters dr：Radius feed per turn，unit mm default0.7
    @param  [in] Default parameters max_t_ms：Maximum exploration time，unit ms default 60000
    @param  [in] Default parameters max_vel：Maximum linear velocity，unit mm/s default 5
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def FT_SpiralSearch(self, rcs, ft, dr=0.7, max_t_ms=60000, max_vel=5):
        rcs = int(rcs)
        ft = float(ft)
        dr = float(dr)
        max_t_ms = float(max_t_ms)
        max_vel = float(max_vel)
        error = self.robot.FT_SpiralSearch(rcs, ft, dr, max_t_ms, max_vel)
        return error

    """   
    @brief  Rotate insert
    @param  [in] Required parameters rcs Reference coordinate system，0-Tool coordinate system，1-Base coordinate system
    @param  [in] Required parameters ft：Force or torque threshold (0~100)，unit N or Nm
    @param  [in] Required parameters orn force/Torque direction，1-alongzAxial direction，2-CirclezAxial direction
    @param  [in] Default parameters angVelRot：Rotation angular velocity，unit °/s  default 3
    @param  [in] Default parameters angleMax：Maximum rotation angle，unit ° default 45
    @param  [in] Default parameters angAccmax：Maximum rotation acceleration，unit °/s^2，Not used yet default0
    @param  [in] Default parameters rotorn：Direction of rotation，1-Clockwise，2-Counterclockwise default1
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def FT_RotInsertion(self, rcs, ft, orn, angVelRot=3, angleMax=45, angAccmax=0, rotorn=1):
        rcs = int(rcs)
        ft = float(ft)
        orn = int(orn)
        angVelRot = float(angVelRot)
        angleMax = float(angleMax)
        angAccmax = float(angAccmax)
        rotorn = int(rotorn)
        error = self.robot.FT_RotInsertion(rcs, angVelRot, ft, angleMax, orn, angAccmax, rotorn)
        return error

    """   
    @brief  Linear insertion
    @param  [in] Required parameters rcs Reference coordinate system，0-Tool coordinate system，1-Base coordinate system
    @param  [in] Required parameters ft：Force or torque threshold (0~100)，unit N or Nm
    @param  [in] Required parameters disMax：Maximum insertion distance，unit mm
    @param  [in] Required parameters linorn：Insert direction:0-Negative direction，1-Positive direction
    @param  [in] Default parameters lin_v：Linear speed，unit mm/s default1
    @param  [in] Default parameters lin_a：Linear acceleration，unit mm/s^2，Not used yet default1
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def FT_LinInsertion(self, rcs, ft, disMax, linorn, lin_v=1.0, lin_a=1.0):
        rcs = int(rcs)
        ft = float(ft)
        disMax = float(disMax)
        linorn = int(linorn)
        lin_v = float(lin_v)
        lin_a = float(lin_a)
        error = self.robot.FT_LinInsertion(rcs, ft, lin_v, lin_a, disMax, linorn)
        return error

    """   
    @brief  Calculate the intermediate plane position to start
    @param  [in] NULL
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def FT_CalCenterStart(self):
        error = self.robot.FT_CalCenterStart()
        return error

    """   
    @brief  Calculate the position of the intermediate plane ends
    @param  [in] NULL
    @return Error code success- 0, fail-Error code
    @return Return value（The call returns successfully）pos=[x,y,z,rx,ry,rz]
    """

    @log_call
    @xmlrpc_timeout
    def FT_CalCenterEnd(self):
        _error = self.robot.FT_CalCenterEnd()
        error = _error[0]
        if error == 0:
            return error, [_error[1], _error[2], _error[3], _error[4], _error[5], _error[6]]
        else:
            return error

    """   
    @brief  Surface positioning
    @param  [in] Required parameters rcs：Reference coordinate system，0-Tool coordinate system，1-Base coordinate system
    @param  [in] Required parameters dir：Movement direction，1-Positive direction，2-Negative direction
    @param  [in] Required parameters axis：Move the axis，1-x，2-y，3-z
    @param  [in] Required parameters disMax：Maximum Exploration Distance，unit mm
    @param  [in] Required parameters ft：Action termination force threshold，unit N
    @param  [in] Default parameters lin_v：Explore straight-line speed，unit mm/s default3
    @param  [in] Default parameters lin_a：Explore linear acceleration，unit mm/s^2 default0
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def FT_FindSurface(self, rcs, dir, axis, disMax, ft, lin_v=3.0, lin_a=0.0):
        rcs = int(rcs)
        dir = int(dir)
        axis = int(axis)
        ft = float(ft)
        lin_v = float(lin_v)
        lin_a = float(lin_a)
        error = self.robot.FT_FindSurface(rcs, dir, axis, lin_v, lin_a, disMax, ft)
        return error

    """   
    @brief  Flexible control off
    @param  [in] NULL
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def FT_ComplianceStop(self):
        error = self.robot.FT_ComplianceStop()
        return error

    """   
    @brief  Flexible control is turned on
    @param  [in] Required parameters p: Position adjustment coefficient or flexibility coefficient
    @param  [in] Required parameters force：Soft opening force threshold，unit N
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def FT_ComplianceStart(self, p, force):
        p = float(p)
        force = float(force)
        error = self.robot.FT_ComplianceStart(p, force)
        return error

    """   
    @brief  Load identification filtering initialization
    @param  [in] NULL
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def LoadIdentifyDynFilterInit(self):
        error = self.robot.LoadIdentifyDynFilterInit()
        return error

    """   
    @brief  Load identification variable initialization
    @param  [in] NULL
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def LoadIdentifyDynVarInit(self):
        error = self.robot.LoadIdentifyDynVarInit()
        return error

    """   
    @brief  Load identification main program
    @param  [in] Required parameters joint_torque Joint torque j1-j6
    @param  [in] Required parameters joint_pos Joint position j1-j6
    @param  [in] Required parameters t Sampling cycle
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def LoadIdentifyMain(self, joint_torque, joint_pos, t):
        joint_torque = list(map(float, joint_torque))
        joint_pos = list(map(float, joint_pos))
        t = float(t)
        error = self.robot.LoadIdentifyMain(joint_torque, joint_pos, t)
        return error

    """   
    @brief  Get load identification results
    @param  [in] Required parameters gain Gravity coefficientdouble[6]，Centrifugal term coefficientdouble[6] 
    @return Error code success- 0, fail-Error code
    @return Return value（The call returns successfully）weight Load weight
    @return Return value（The call returns successfully）cog Load centroid [x,y,z]
    """

    @log_call
    @xmlrpc_timeout
    def LoadIdentifyGetResult(self, gain):
        gain = list(map(float, gain))
        _error = self.robot.LoadIdentifyGetResult(gain)
        error = _error[0]
        if error == 0:
            return error, _error[1], [_error[2], _error[3], _error[4]]
        else:
            return error

    """   
    ***************************************************************************Conveyor belt function********************************************************************************************
    """

    """   
    @brief  Drive belt start、stop
    @param  [in] Required parameters status state，1-start up，0-stop 
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def ConveyorStartEnd(self, status):
        if self.GetSafetyCode() != 0:
            return self.GetSafetyCode()
        status = int(status)
        error = self.robot.ConveyorStartEnd(status)
        return error

    """   
    @brief  RecordIODetection point
    @param  [in] NULL
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def ConveyorPointIORecord(self):
        error = self.robot.ConveyorPointIORecord()
        return error

    """   
    @brief  RecordApoint
    @param  [in] NULL
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def ConveyorPointARecord(self):
        error = self.robot.ConveyorPointARecord()
        return error

    """   
    @brief  Record the reference point
    @param  [in] NULL
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def ConveyorRefPointRecord(self):
        error = self.robot.ConveyorRefPointRecord()
        return error

    """   
    @brief  RecordBpoint
    @param  [in] NULL
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def ConveyorPointBRecord(self):
        error = self.robot.ConveyorPointBRecord()
        return error

    """   
    @brief  Conveyor belt workpieceIOTest
    @param  [in] Required parameters max_t Maximum detection time，unitms
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def ConveyorIODetect(self, max_t):
        max_t = int(max_t)
        error = self.robot.ConveyorIODetect(max_t)
        return error

    """   
    @brief  Get the current position of the object
    @param  [in] Required parameters  mode 1-Tracking and crawling 2-Tracking movement 3-TPDtrack
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def ConveyorGetTrackData(self, mode):
        mode = int(mode)
        error = self.robot.ConveyorGetTrackData(mode)
        return error

    """   
    @brief  Drive belt tracking start
    @param  [in] Required parameters  status state，1-start up，0-stop
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def ConveyorTrackStart(self, status):
        status = int(status)
        error = self.robot.ConveyorTrackStart(status)
        return error

    """   
    @brief  Drive belt tracking stop
    @param  [in] NULL
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def ConveyorTrackEnd(self):
        error = self.robot.ConveyorTrackEnd()
        return error

    """   
    @brief  Drive belt parameter configuration
    @param  [in] Required parameters  param = [encChannel,resolution,lead,wpAxis,vision,speedRadio]  encChannelEncoder channel 1-2,resolution Encoder resolution Number of pulses of encoder rotation,
    leadMechanical transmission ratio The encoder rotates one circle and the conveyor belt moves,wpAxis  Workpiece coordinate system number Select the workpiece coordinate system number for tracking motion function，Tracking and crawling、TPDSet tracking to0,vision Is it suitable for vision  0 Not worthy  1 match,
    speedRadio Speed ​​ratio  Tracking and grab speed range for conveyor belts is（1-100）  Tracking movement、TPDTracking is set to1
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def ConveyorSetParam(self, param):
        param = list(map(float, param))
        error = self.robot.ConveyorSetParam(param)
        return error

    """   
    @brief  Transmission belt grab point compensation
    @param  [in] Required parameters cmp Compensation location [x,y,z]
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def ConveyorCatchPointComp(self, cmp):
        cmp = list(map(float, cmp))
        error = self.robot.ConveyorCatchPointComp(cmp)
        return error

    """   
    @brief  Linear motion
    @param  [in] Required parameters  name cvrCatchPointandcvrRaisePoint
    @param  [in] Required parameters tool Tool number
    @param  [in] Required parameters wobj Workpiece number
    @param  [in] Default parameters vel speed default20
    @param  [in] Default parameters acc Acceleration default100
    @param  [in] Default parameters ovl Speed ​​scaling factor default100
    @param  [in] Default parameters blendR:[-1.0]-Exercise in place (block)，[0~1000]-Smooth radius (Non-blocking)，unit [mm] default-1.0
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def ConveyorTrackMoveL(self, name, tool, wobj, vel=20, acc=100, ovl=100, blendR=-1.0):
        if self.GetSafetyCode() != 0:
            return self.GetSafetyCode()
        name = str(name)
        tool = int(tool)
        wobj = int(wobj)
        vel = float(vel)
        acc = float(acc)
        ovl = float(ovl)
        blendR = float(blendR)
        error = self.robot.ConveyorTrackMoveL(name, tool, wobj, vel, acc, ovl, blendR, 0, 0)
        return error

    """   
    ***************************************************************************Welding function********************************************************************************************
    """


    """   
    @brief  Welding begins 
    @param  [in] Required parameters ioType iotype 0-ControllerIO； 1-ExtendedIO
    @param  [in] Required parameters arcNum Welding machine configuration file number
    @param  [in] Required parameters timeout Arc timeout time
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def ARCStart(self, ioType, arcNum, timeout):
        ioType = int(ioType)
        arcNum = int(arcNum)
        timeout = int(timeout)
        error = self.robot.ARCStart(ioType, arcNum, timeout)
        return error

    """   
    @brief  The welding ends 
    @param  [in] Required parameters ioType iotype 0-ControllerIO； 1-ExtendedIO
    @param  [in] Required parameters arcNum Welding machine configuration file number
    @param  [in] Required parameters timeout Arc expiration timeout time
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def ARCEnd(self, ioType, arcNum, timeout):
        ioType = int(ioType)
        arcNum = int(arcNum)
        timeout = int(timeout)
        error = self.robot.ARCEnd(ioType, arcNum, timeout)
        return error

    """   
    @brief  Set the correspondence between welding current and output analog quantity 
    @param  [in] Required parameters currentMin Welding current-Analog output linear relationship current value on the left point(A)
    @param  [in] Required parameters currentMax Welding current-Analog output linear relationship current value on the right point(A)
    @param  [in] Required parameters outputVoltageMin Welding current-Analog output linear relationship analog output voltage value(V)
    @param  [in] Required parameters outputVoltageMax Welding current-Analog output linear relationship point on the right point of analog output voltage(V)
    @param  [in] Required parameters AOIndex Welding current analog output port
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def WeldingSetCurrentRelation(self, currentMin, currentMax, outputVoltageMin, outputVoltageMax,AOIndex):
        currentMin = float(currentMin)
        currentMax = float(currentMax)
        outputVoltageMin = float(outputVoltageMin)
        outputVoltageMax = float(outputVoltageMax)
        AOIndex =int(AOIndex)
        error = self.robot.WeldingSetCurrentRelation(currentMin, currentMax, outputVoltageMin, outputVoltageMax,AOIndex)
        return error

    """   
    @brief  Set the correspondence between welding voltage and output analog quantity 
    @param  [in] Required parameters weldVoltageMin Welding voltage-Analog output linear relationship left point welding voltage value(A)
    @param  [in] Required parameters weldVoltageMax Welding voltage-Analog output linear relationship point welding voltage value(A)
    @param  [in] Required parameters outputVoltageMin Welding voltage-Analog output linear relationship analog output voltage value(V)
    @param  [in] Required parameters outputVoltageMax Welding voltage-Analog output linear relationship point on the right point of analog output voltage(V)    
    @param  [in] Required parameters AOIndex Welding voltage analog output port
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def WeldingSetVoltageRelation(self, weldVoltageMin, weldVoltageMax, outputVoltageMin, outputVoltageMax,AOIndex):
        weldVoltageMin = float(weldVoltageMin)
        weldVoltageMax = float(weldVoltageMax)
        outputVoltageMin = float(outputVoltageMin)
        outputVoltageMax = float(outputVoltageMax)
        AOIndex =int(AOIndex)

        error = self.robot.WeldingSetVoltageRelation(weldVoltageMin, weldVoltageMax, outputVoltageMin, outputVoltageMax,AOIndex)
        return error

    """   
    @brief  Obtain the correspondence between welding current and output analog quantity 
    @return Error code success- 0, fail-Error code
    @return Return value（The call returns successfully）currentMin Welding current-Analog output linear relationship current value on the left point(A)
    @return Return value（The call returns successfully）currentMax Welding current-Analog output linear relationship current value on the right point(A)
    @return Return value（The call returns successfully）outputVoltageMin Welding current-Analog output linear relationship analog output voltage value(V)
    @return Return value（The call returns successfully）outputVoltageMax Welding current-Analog output linear relationship point on the right point of analog output voltage(V)
    @return Return value（The call returns successfully） AOIndex Welding voltage and current analog output port
    """

    @log_call
    @xmlrpc_timeout
    def WeldingGetCurrentRelation(self):

        try:
            _error = self.robot.WeldingGetCurrentRelation()
            error = _error[0]
            if error == 0:
                return error, _error[1], _error[2], _error[3], _error[4], _error[5]
            return _error
        except Exception as e:
            return RobotError.ERR_RPC_ERROR

    """   
    @brief  Obtain the correspondence between welding voltage and output analog quantity 
    @return Error code success- 0, fail-Error code
    @return Return value（The call returns successfully）weldVoltageMin Welding voltage-Analog output linear relationship left point welding voltage value(A)
    @return Return value（The call returns successfully）weldVoltageMax Welding voltage-Analog output linear relationship point welding voltage value(A)
    @return Return value（The call returns successfully）outputVoltageMin Welding voltage-Analog output linear relationship analog output voltage value(V)
    @return Return value（The call returns successfully）outputVoltageMax Welding current-Analog output linear relationship point on the right point of analog output voltage(V)
    @return Return value（The call returns successfully） AOIndex Welding voltage analog output port
    """

    @log_call
    @xmlrpc_timeout
    def WeldingGetVoltageRelation(self):

        try:
            _error = self.robot.WeldingGetVoltageRelation()
            error = _error[0]
            if error == 0:
                return error, _error[1], _error[2], _error[3], _error[4], _error[5]
            return _error
        except Exception as e:
            return RobotError.ERR_RPC_ERROR

    """   
    @brief  Set welding current 
    @param  [in] Required parameters ioType iotype 0-ControllerIO； 1-ExtendedIO
    @param  [in] Required parameters float current Welding current value(A)
    @param  [in] Required parameters AOIndex Welding current control box analog output port(0-1)
    @param  [in] Required parameters blend Is it smooth or not 0-Non-smooth，1-smooth
    @return Error code success- 0, fail-
    """

    @log_call
    @xmlrpc_timeout
    def WeldingSetCurrent(self, ioType, current, AOIndex,blend):
        ioType = int(ioType)
        current = float(current)
        AOIndex = int(AOIndex)
        blend = int(blend)
        error = self.robot.WeldingSetCurrent(ioType, current, AOIndex,blend)
        return error

    """   
    @brief  Set the welding voltage 
    @param  [in] Required parameters ioType iotype 0-ControllerIO； 1-ExtendedIO
    @param  [in] Required parameters float voltage Welding voltage value(A)
    @param  [in] Required parameters AOIndex Welding voltage control box analog output port(0-1)
    @param  [in] Required parameters blend Is it smooth or not 0-Non-smooth，1-smooth
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def WeldingSetVoltage(self, ioType, voltage, AOIndex,blend):
        ioType = int(ioType)
        voltage = float(voltage)
        AOIndex = int(AOIndex)
        blend = int(blend)
        error = self.robot.WeldingSetVoltage(ioType, voltage, AOIndex,blend)
        return error

    """   
    @brief  Set swing parameters 
    @param  [in] Required parameters int weaveNum Parameter configuration number of the slanting welding
    @param  [in] Required parameters int weaveType Swing type 0-Planar triangle wave swing；1-verticalLTriangular wave swing；2-Clockwise circular swing；3-Counterclockwise circular swing；4-Planar sine wave swing；5-verticalLSine wave swing；6-Vertical triangle wave swing；7-Vertical sine wave swing
    @param  [in] Required parameters float weaveFrequency Swing frequency(Hz)
    @param  [in] Required parameters int weaveIncStayTime Waiting mode 0-The cycle does not include waiting time；1-The cycle includes waiting time
    @param  [in] Required parameters float weaveRange Swing amplitude(mm)
    @param  [in] Required parameters weaveLeftRange Vertical triangle swing left chord length(mm)
    @param  [in] Required parameters weaveRightRange Vertical triangle swing right chord length(mm)
    @param  [in] Required parameters additionalStayTime Vertical triangle swing vertical triangle point residence time(mm)
    @param  [in] Required parameters int weaveLeftStayTime Swing left residence time(ms)
    @param  [in] Required parameters int weaveRightStayTime Swing right stay time(ms)
    @param  [in] Required parameters int weaveCircleRadio Circular swing-Callback ratio(0-100%)
    @param  [in] Required parameters int weaveStationary Wait for swing position，0-Waiting for the time to move the position；1-The position is still in the waiting time
    @param  [in] Required parameters float weaveYawAngle Swing direction azimuth angle（Swing aroundZAxis rotation），unit°,default0
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def WeaveSetPara(self, weaveNum, weaveType, weaveFrequency, weaveIncStayTime, weaveRange,
                     weaveLeftRange, weaveRightRange, additionalStayTime, weaveLeftStayTime,
                     weaveRightStayTime, weaveCircleRadio, weaveStationary,weaveYawAngle=0):
        weaveNum = int(weaveNum)
        weaveType = int(weaveType)
        weaveFrequency = float(weaveFrequency)
        weaveIncStayTime = int(weaveIncStayTime)
        weaveRange = float(weaveRange)
        weaveLeftRange = float(weaveLeftRange)
        weaveRightRange = float(weaveRightRange)
        additionalStayTime = int(additionalStayTime)
        weaveLeftStayTime = int(weaveLeftStayTime)
        weaveRightStayTime = int(weaveRightStayTime)
        weaveCircleRadio = int(weaveCircleRadio)
        weaveStationary = int(weaveStationary)
        weaveYawAngle = float(weaveYawAngle)
        error = self.robot.WeaveSetPara(weaveNum, weaveType, weaveFrequency, weaveIncStayTime, weaveRange,
                                        weaveLeftRange, weaveRightRange, additionalStayTime,
                                        weaveLeftStayTime, weaveRightStayTime, weaveCircleRadio, weaveStationary,weaveYawAngle)
        return error

    """   
    @brief  Set swing parameters instantly 
    @param  [in] Required parameters int weaveNum Parameter configuration number of the slanting welding
    @param  [in] Required parameters int weaveType Swing type 0-Planar triangle wave swing；1-verticalLTriangular wave swing；2-Clockwise circular swing；3-Counterclockwise circular swing；4-Planar sine wave swing；5-verticalLSine wave swing；6-Vertical triangle wave swing；7-
    @param  [in] Required parameters float weaveFrequency Swing frequency(Hz)
    @param  [in] Required parameters int weaveIncStayTime Waiting mode 0-The cycle does not include waiting time；1-The cycle includes waiting time
    @param  [in] Required parameters float weaveRange Swing amplitude(mm)
    @param  [in] Required parameters int weaveLeftStayTime Swing left residence time(ms)
    @param  [in] Required parameters int weaveRightStayTime Swing right stay time(ms)
    @param  [in] Required parameters int weaveCircleRadio Circular swing-Callback ratio(0-100%)
    @param  [in] Required parameters int weaveStationary Wait for swing position，0-Waiting for the time to move the position；1-The position is still in the waiting time
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def WeaveOnlineSetPara(self, weaveNum, weaveType, weaveFrequency, weaveIncStayTime, weaveRange, weaveLeftStayTime,
                           weaveRightStayTime, weaveCircleRadio, weaveStationary):
        weaveNum = int(weaveNum)
        weaveType = int(weaveType)
        weaveFrequency = float(weaveFrequency)
        weaveIncStayTime = int(weaveIncStayTime)
        weaveRange = float(weaveRange)
        weaveLeftStayTime = int(weaveLeftStayTime)
        weaveRightStayTime = int(weaveRightStayTime)
        weaveCircleRadio = int(weaveCircleRadio)
        weaveStationary = int(weaveStationary)
        try:
            error = self.robot.WeaveOnlineSetPara(weaveNum, weaveType, weaveFrequency, weaveIncStayTime, weaveRange,
                                                  weaveLeftStayTime, weaveRightStayTime, weaveCircleRadio,
                                                  weaveStationary)
            return error
        except Exception as e:
            return RobotError.ERR_RPC_ERROR

    """   
    @brief  The swing begins 
    @param  [in] Required parameters int weaveNum Parameter configuration number of the slanting welding
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def WeaveStart(self, weaveNum):
        weaveNum = int(weaveNum)
        try:
            error = self.robot.WeaveStart(weaveNum)
            return error
        except Exception as e:
            return RobotError.ERR_RPC_ERROR

    """   
    @brief  The swing ends 
    @param  [in] Required parameters int weaveNum Parameter configuration number of the slanting welding
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def WeaveEnd(self, weaveNum):
        weaveNum = int(weaveNum)
        try:
            error = self.robot.WeaveEnd(weaveNum)
            return error
        except Exception as e:
            return RobotError.ERR_RPC_ERROR

    """   
    @brief  Forward wire delivery 
    @param  [in] Required parameters int ioType iotype  0-ControllerIO；1-ExtendedIO
    @param  [in] Required parameters int wireFeed Wire feeding control  0-Stop sending silk；1-Shield delivery
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def SetForwardWireFeed(self, ioType, wireFeed):
        ioType = int(ioType)
        wireFeed = int(wireFeed)
        try:
            error = self.robot.SetForwardWireFeed(ioType, wireFeed)
            return error
        except Exception as e:
            return RobotError.ERR_RPC_ERROR

    """   
    @brief  Reverse wire feed 
    @param  [in] Required parameters int ioType iotype  0-ControllerIO；1-ExtendedIO
    @param  [in] Required parameters int wireFeed Wire feeding control  0-Stop sending silk；1-Shield delivery
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def SetReverseWireFeed(self, ioType, wireFeed):
        ioType = int(ioType)
        wireFeed = int(wireFeed)
        try:
            error = self.robot.SetReverseWireFeed(ioType, wireFeed)
            return error
        except Exception as e:
            return RobotError.ERR_RPC_ERROR

    """   
    @brief  aspirated
    @param  [in] Required parameters int ioType iotype  0-ControllerIO；1-ExtendedIO
    @param  [in] Required parameters int airControl Air supply control  0-Stop air delivery；1-aspirated
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def SetAspirated(self, ioType, airControl):
        ioType = int(ioType)
        airControl = int(airControl)
        try:
            error = self.robot.SetAspirated(ioType, airControl)
            return error
        except Exception as e:
            return RobotError.ERR_RPC_ERROR


    """   
    @brief  Segment welding obtains position and posture
    @param  [in]Required parameters startPos=[x,y,z,rx,ry,rz] Start point coordinates
    @param  [in]Required parameters endPos=[x,y,z,rx,ry,rz] Terminal point coordinates
    @param  [in]Required parameters startDistance Length from welding point to starting point
    @return Error code success- 0, fail-Error code    
    @return Return value（The call returns successfully） weldPointDesc=[x,y,z,rx,ry,rz] Cartesian coordinate information of welding points
    @return Return value（The call returns successfully） weldPointJoint=[j1,j2,j3,j4,j5,j6] Joint coordinate information of welding points
    @return Return value（The call returns successfully） tool Tool number
    @return Return value（The call returns successfully） user Workpiece number
    """

    @log_call
    @xmlrpc_timeout
    def GetSegmentWeldPoint(self, startPos, endPos, startDistance):
        startPos = list(map(float, startPos))
        endPos = list(map(float, endPos))
        startDistance = float(startDistance)
        _error = self.robot.GetSegmentWeldPoint(startPos, endPos, startDistance)
        error = _error[0]
        if error == 0:
            data = _error[1].split(',')
            if len(data) != 14:
                self.log_error("GetSegmentWeldPoint fail")
                return -1
            else:
                data = list(map(float,data))
                tool = int(data[12])
                work = int(data[13])
                return (error, [ data[0],data[1],data[3],data[4],data[4],data[5]],
                        [data[6],data[7],data[8],data[9],data[10],data[11]],tool, work)
        else:
            return error

    """   
    @brief  Segmented welding start
    @param  [in] Required parameters startDesePos: Initial Cartesian Position，unit [mm][°]
    @param  [in] Required parameters endDesePos: Target Cartesian pose，unit [mm][°]
    @param  [in] Required parameters startJPos: Target joint position，unit [°]
    @param  [in] Required parameters endJPos: Target joint position，unit [°] 
    @param  [in] Required parameters weldLength: Welding length，unit [mm]
    @param  [in] Required parameters noWeldLength: Non-welded length，unit [mm]    
    @param  [in] Required parameters weaveType Swing type 0-Planar triangle wave swing；1-verticalLTriangular wave swing；2-Clockwise circular swing；3-Counterclockwise circular swing；4-Planar sine wave swing；5-verticalLSine wave swing；6-Vertical triangle wave swing；7-Vertical sine wave swing
    @param  [in] Required parameters arcNum Welding machine configuration file number
    @param  [in] Required parameters timeout Arc expiration timeout time
    @param  [in] Required parameters isWeave true-welding false-No soldering
    @param  [in] Required parameters int weaveNum Parameter configuration number of the slanting welding
    @param  [in] Required parameters tool: Tool number，[0~14]
    @param  [in] Required parameters tool: Tool number，[0~14]
    @param  [in] Required parameters user: Workpiece number，[0~14]
    @param  [in] Default parameters vel：Speed ​​percentage，[0~100] default20.0
    @param  [in] Default parameters acc：Acceleration percentage，[0~100] Not open yet default0.0
    @param  [in] Default parameters ovl: Speed ​​scaling factor，[0~100] default100.0
    @param  [in] Default parameters blendR:[-1.0]-Exercise in place (block)，[0~1000]-Smooth radius (Non-blocking)，unit [mm] default-1.0
    @param  [in] Default parameters exaxis_pos: External axis 1 Location ~ External axis 4 Location default[0.0,0.0,0.0,0.0]
    @param  [in] Default parameters search:[0]-No welding wire position search，[1]-Wire positioning
    @param  [in] Default parameters offset_flag:[0]-No offset，[1]-Workpieces/Offset under base coordinate system，[2]-Offset under tool coordinate system default 0
    @param  [in] Default parameters offset_pos: Pose offset，unit [mm][°] default[0.0,0.0,0.0,0.0,0.0,0.0]
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def SegmentWeldStart(self, startDesePos, endDesePos, startJPos, endJPos, weldLength, noWeldLength, weldIOType,
                         arcNum, weldTimeout, isWeave, weaveNum, tool, user,
                         vel=20.0, acc=0.0, ovl=100.0, blendR=-1.0, exaxis_pos=[0.0, 0.0, 0.0, 0.0], search=0,
                         offset_flag=0, offset_pos=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]):
        startDesePos = list(map(float, startDesePos))
        endDesePos = list(map(float, endDesePos))
        startJPos = list(map(float, startJPos))
        endJPos = list(map(float, endJPos))

        weldLength = float(weldLength)
        noWeldLength = float(noWeldLength)
        weldIOType = int(weldIOType)
        arcNum = int(arcNum)
        weldTimeout = int(weldTimeout)
        isWeave = bool(isWeave)
        weaveNum = int(weaveNum)
        tool = int(tool)
        user = int(user)
        vel = float(vel)
        acc = float(acc)
        ovl = float(ovl)
        blendR = float(blendR)
        exaxis_pos = list(map(float, exaxis_pos))
        search = int(search)
        offset_flag = int(offset_flag)
        offset_pos = list(map(float, offset_pos))

        rtn = 0
        # Get the distance between the starting point and the end point and the angle cosine value of each direction
        # print("1",startDesePos,endDesePos)
        result = self.robot.GetSegWeldDisDir(startDesePos[0], startDesePos[1], startDesePos[2], endDesePos[0],
                                             endDesePos[1], endDesePos[2])
        # print("result",result)
        if result[0] != 0:
            return int(result[0])

        distance = result[1]
        endOffPos = list(offset_pos)

        rtn = self.robot.MoveJ(startJPos, startDesePos, tool, user, vel, acc, ovl, exaxis_pos, blendR, offset_flag,
                               offset_pos)
        # print("rtn1", rtn)
        if rtn != 0:
            return rtn

        weldNum = 0
        noWeldNum = 0
        i = 0
        while i < int(distance / (weldLength + noWeldLength)) * 2 + 2:
            if i % 2 == 0:
                weldNum += 1
                if weldNum * weldLength + noWeldNum * noWeldLength > distance:

                    rtn = self.robot.ARCStart(weldIOType, arcNum, weldTimeout)
                    # print("rtn2", rtn)
                    if rtn != 0:
                        return rtn
                    if isWeave:
                        rtn = self.robot.WeaveStart(weaveNum)
                        if rtn != 0:
                            # print("rtn3", rtn)
                            return rtn

                    # getsegmentrtn = self.robot.GetSegmentWeldPoint(startDesePos,endDesePos,weldNum* weldLength + noWeldNum * noWeldLength)
                    # # print("getsegmentrtn", getsegmentrtn)
                    # # print(startDesePos, endDesePos, weldNum * weldLength + noWeldNum * noWeldLength)
                    # # print("weldNum", weldNum, "weldLength", weldLength)
                    # # print("noWeldNum", noWeldNum, "noWeldLength", noWeldLength)
                    # if getsegmentrtn[0] != 0  :
                    #     return getsegmentrtn[0]
                    # data = getsegmentrtn[1].split(',')
                    # data = list(map(float, data))
                    # if len(data) != 14:
                    #     self.log_error("GetSegmentWeldPoint fail")
                    #     return -1
                    # tmpJoint = [data[0],data[1],data[2],data[3],data[4],data[5]]
                    # tmpWeldDesc = [data[6],data[7],data[8],data[9],data[10],data[11]]
                    # tmpTool = int(data[12])
                    # tmpUser = int(data[13])
                    rtn = self.robot.MoveL(endJPos,endDesePos, tool, user, vel, acc, ovl, blendR, exaxis_pos,
                                           search, 0, endOffPos)
                    # print("rtn3", rtn,endJPos,endDesePos)
                    if rtn != 0:
                        self.robot.ARCEnd(weldIOType, arcNum, weldTimeout)
                        if isWeave:
                            rtn = self.robot.WeaveEnd(weaveNum)
                            # print("rtn4", rtn)
                            if rtn != 0:
                                return rtn
                        return rtn
                    rtn = self.robot.ARCEnd(weldIOType, arcNum, weldTimeout)
                    # print("rtn5", rtn)
                    if rtn != 0:
                        break
                    if isWeave:
                        rtn = self.robot.WeaveEnd(weaveNum)
                        # print("rtn6", rtn)
                        if rtn != 0:
                            break

                else:
                    rtn = self.robot.ARCStart(weldIOType, arcNum, weldTimeout)
                    # print("rtn7", rtn)
                    if rtn != 0:
                        return rtn
                    if isWeave:
                        rtn = self.robot.WeaveStart(weaveNum)
                        # print("rtn8", rtn)
                        if rtn != 0:
                            return rtn

                    getsegmentrtn = self.robot.GetSegmentWeldPoint(startDesePos, endDesePos,
                                                                   weldNum * weldLength + noWeldNum * noWeldLength)
                    # print("rtn9", getsegmentrtn)
                    # print(startDesePos, endDesePos, weldNum * weldLength + noWeldNum * noWeldLength)
                    # print("weldNum", weldNum, "weldLength", weldLength)
                    # print("noWeldNum", noWeldNum, "noWeldLength", noWeldLength)
                    if getsegmentrtn[0] != 0:
                        return getsegmentrtn[0]
                    data = getsegmentrtn[1].split(',')
                    data = list(map(float, data))
                    if len(data) != 14:
                        self.log_error("GetSegmentWeldPoint fail")
                        return -1
                    tmpJoint = [data[0], data[1], data[2], data[3], data[4], data[5]]
                    tmpWeldDesc = [data[6], data[7], data[8], data[9], data[10], data[11]]
                    tmpTool = int(data[12])
                    tmpUser = int(data[13])
                    # print("tmpJoint",tmpJoint,tmpWeldDesc,tmpTool,tmpUser)
                    time.sleep(1)
                    nihao = self.robot.MoveL(tmpJoint, tmpWeldDesc, tmpTool, tmpUser, vel, acc, ovl, blendR, exaxis_pos,
                                           search, 0, endOffPos)
                    # print("rtn10nihao", nihao)
                    if nihao != 0:
                        self.robot.ARCEnd(weldIOType, arcNum, weldTimeout)
                        if isWeave:
                            rtn = self.robot.WeaveEnd(weaveNum)
                            # print("rtn11", rtn)
                            if rtn != 0:
                                return rtn
                        return rtn
                    rtn = self.robot.ARCEnd(weldIOType, arcNum, weldTimeout)
                    # print("rtn12", rtn)
                    if rtn != 0:
                        return rtn
                    if isWeave:
                        rtn = self.robot.WeaveEnd(weaveNum)
                        # print("rtn13", rtn)
                        if rtn != 0:
                            return rtn
            else:
                noWeldNum += 1
                if weldNum * weldLength + noWeldNum * noWeldLength > distance:
                    # getsegmentrtn = self.robot.GetSegmentWeldPoint(startDesePos, endDesePos, weldNum* weldLength + noWeldNum * noWeldLength)
                    # # print("rtn14", getsegmentrtn)
                    # # print(startDesePos, endDesePos, weldNum * weldLength + noWeldNum * noWeldLength)
                    # # print("weldNum", weldNum, "weldLength", weldLength)
                    # # print("noWeldNum", noWeldNum, "noWeldLength", noWeldLength)
                    # if getsegmentrtn[0] != 0:
                    #     return getsegmentrtn[0]
                    # data = getsegmentrtn[1].split(',')
                    # data = list(map(float,data))
                    # if len(data) != 14:
                    #     self.log_error("GetSegmentWeldPoint fail")
                    #     return -1
                    # tmpJoint = [data[0], data[1], data[2], data[3], data[4], data[5]]
                    # tmpWeldDesc = [data[6], data[7], data[8], data[9], data[10], data[11]]
                    # tmpTool = int(data[12])
                    # tmpUser = int(data[13])
                    rtn = self.robot.MoveL(endJPos,endDesePos, tool, user, vel, acc, ovl, blendR, exaxis_pos,
                                           search, 0, endOffPos)
                    # print("rtn15", rtn,endJPos,endDesePos)
                    if rtn != 0:
                       return rtn
                    break
                else:
                    getsegmentrtn = self.robot.GetSegmentWeldPoint(startDesePos, endDesePos, weldNum* weldLength + noWeldNum * noWeldLength)
                    # print("rtn16", getsegmentrtn,startDesePos,endDesePos,weldNum* weldLength + noWeldNum * noWeldLength)

                    # print(startDesePos,endDesePos,weldNum* weldLength + noWeldNum * noWeldLength)
                    # print("weldNum",weldNum,"weldLength",weldLength)
                    # print("noWeldNum", noWeldNum, "noWeldLength", noWeldLength)
                    if getsegmentrtn[0] != 0:
                        return getsegmentrtn[0]
                    data = getsegmentrtn[1].split(',')
                    data = list(map(float, data))
                    if len(data) != 14:
                        self.log_error("GetSegmentWeldPoint fail")
                        return -1
                    tmpJoint = [data[0], data[1], data[2], data[3], data[4], data[5]]
                    tmpWeldDesc = [data[6], data[7], data[8], data[9], data[10], data[11]]
                    tmpTool = int(data[12])
                    tmpUser = int(data[13])
                    rtn = self.robot.MoveL(tmpJoint, tmpWeldDesc, tmpTool, tmpUser, vel, acc, ovl, blendR, exaxis_pos,
                                           search, 0, endOffPos)
                    # print("rtn17", rtn)
                    if rtn != 0:
                        return rtn
            i =i + 1
        return rtn

    """   
    @brief  Segmented welding termination
    @param  [in] Required parameters ioType：iotype 0-ControllerIO； 1-ExtendedIO
    @param  [in] Required parameters arcNum：Welding machine configuration file number
    @param  [in] Required parameters timeout：Arc expiration timeout time
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def SegmentWeldEnd(self, ioType, arcNum, timeout):
        ioType = int(ioType)
        arcNum = int(arcNum)
        timeout = int(timeout)

        rtn = self.robot.SegmentWeldEnd(ioType, arcNum, timeout)
        return rtn

    """   
    @brief  Initialize log parameters
    @param  [in]Default parameters output_model：Output mode，0-Direct output；1-Buffered output；2-Asynchronous output，default1
    @param  [in]Default parameters file_path： File saving path+name，The name must bexxx.logForm of，for example/home/fr/linux/fairino.log。
                    The default path to execute the program，Default namefairino_ year+month+data.log(like:fairino_2024_03_13.log);
    @param  [in]Default parameters file_num：Number of files to be stored in scroll，1~20indivual，The default value is5。Single file cap50M;
    @return Error code success-0  fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def LoggerInit(self, output_model=1, file_path="", file_num=5):
        return self.setup_logging(output_model, file_path, file_num)

    """   
    @brief  Set log filtering level
    @param  [in] Default parameters lvl: Filter level value，The smaller the value, the fewer the output logs, 1-error, 2-warnning, 3-inform, 4-debug,The default value is1.
    @return Error code success-0  fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def SetLoggerLevel(self, lvl=1):
        lvl=int(lvl)
        log_level = self.set_log_level(lvl)
        return 0

    """   
    @brief  Download the point table database
    @param  [in] pointTableName Name of the point table to download    pointTable1.db
    @param  [in] saveFilePath The storage path of the download point table   C://test/
    @return Error code success-0  fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def PointTableDownLoad(self, point_table_name, save_file_path):
        if not os.path.exists(save_file_path):
            return RobotError.ERR_SAVE_FILE_PATH_NOT_FOUND

        rtn = self.robot.PointTableDownload(point_table_name)
        if rtn == -1:
            return RobotError.ERR_POINTTABLE_NOTFOUND
        elif rtn != 0:
            return rtn
        port = 20011
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.settimeout(2)
        try:
            client.connect((self.ip_address, port))
        except Exception as e:
            client.close()
            return RobotError.ERR_OTHER
        total_buffer = bytearray(1024 * 1024 * 50)  # 50Mb
        total_size = 0
        recv_md5 = ""
        recv_size = 0
        find_head_flag = False
        while True:
            buffer = client.recv(1024)
            length = len(buffer)
            if length < 1:
                return RobotError.ERR_OTHER
            total_buffer[total_size:total_size + len(buffer)] = buffer

            total_size += len(buffer)
            if not find_head_flag and total_size > 4 and total_buffer[:4].decode('utf-8') == "/f/b":
                find_head_flag = True
            # After finding the file header，Extract file size andMD5Verification code。The file size information is in the total data5Arrive12Bytes，MD5The verification code information is located in the total data13Arrive44Bytes。
            if find_head_flag and total_size > 12 + 32:
                recv_size = int(total_buffer[4:12].decode('utf-8'))
                recv_md5 = total_buffer[12:44].decode('utf-8')
            # Receive the entire file and jump out of the loop
            if find_head_flag and total_size == recv_size:
                break
        if total_size == 0:
            return RobotError.ERR_OTHER
        file_buffer = total_buffer[12 + 32:total_size - 4]

        with open(os.path.join(save_file_path, point_table_name), 'wb') as file_writer:
            file_writer.write(file_buffer[:total_size - 16 - 32])

        check_md5 = calculate_file_md5(save_file_path + point_table_name)
        if check_md5 == recv_md5:
            client.send("SUCCESS".encode('utf-8'))
            return 0
        else:
            client.send("FAIL".encode('utf-8'))
            os.remove(os.path.join(save_file_path, point_table_name))
            return RobotError.ERR_OTHER

    """   
    @brief  Upload point table database
    @param  [in] pointTableFilePath The full path name of the upload point table   C://test/pointTable1.db
    @return Error code success-0  fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def PointTableUpLoad(self, point_table_file_path):
        MAX_UPLOAD_FILE_SIZE = 2 * 1024 * 1024;  # The maximum uploaded file is2Mb
        # Determine whether the uploaded file exists
        if not os.path.exists(point_table_file_path):
            return RobotError.ERR_UPLOAD_FILE_NOT_FOUND

        file_info = os.path.getsize(point_table_file_path)
        total_size = file_info + 16 + 32
        if total_size > MAX_UPLOAD_FILE_SIZE:
            print("Files larger than 2 MB are not supported!")
            return -1

        point_table_name = os.path.basename(point_table_file_path)

        rtn = self.robot.PointTableUpload(point_table_name)
        if rtn != 0:
            return rtn

        port = 20010

        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.settimeout(2)

        try:
            client.connect((self.ip_address, port))
        except Exception as e:
            client.close()
            return RobotError.ERR_OTHER

        client.settimeout(2)

        # client.receive_timeout = 2000
        # client.send_timeout = 2000

        send_md5 = calculate_file_md5(point_table_file_path)

        head_data = f"/f/b{total_size:08d}{send_md5}"
        num = client.send(head_data.encode('utf-8'))
        if num < 1:
            return RobotError.ERR_OTHER

        with open(point_table_file_path, 'rb') as fs:
            file_bytes = fs.read()

        num = client.send(file_bytes)
        if num < 1:
            return RobotError.ERR_OTHER
        end_data = "/b/f"
        num = client.send(end_data.encode('utf-8'))
        if num < 1:
            return RobotError.ERR_OTHER

        result_buf = client.recv(1024)
        if result_buf[:7].decode('utf-8') == "SUCCESS":
            return RobotError.ERR_SUCCESS
        else:
            return RobotError.ERR_OTHER

    """   
    @brief  Point table switching
    @param  [in] PointTableSwitch The name of the point table to switch   "pointTable1.db",When the point table is empty，Right now""hour，Indicates thatluaThe program is updated to the initial program where the point table is not applied
    @return Error code success-0   fail-Error code 
    @return mistakeerrorStr
    """

    @log_call
    @xmlrpc_timeout
    def PointTableSwitch(self, point_table_name):
        rtn = self.robot.PointTableSwitch(point_table_name)  # Switch point table
        if rtn != 0:
            if rtn == RobotError.ERR_POINTTABLE_NOTFOUND:
                error_str = "PointTable not Found!"
            else:
                error_str = "PointTable not Found!"
            return rtn, error_str
        return rtn

    """   
    @brief  Point table updateluadocument
    @param  [in] pointTableName The name of the point table to switch   "pointTable1.db",When the point table is empty，Right now""hour，Indicates thatluaThe program is updated to the initial program where the point table is not applied
    @param  [in] luaFileName To be updatedluaFile name   "testPointTable.lua"
    @return Error code success-0   fail-Error code 
    @return mistakeerrorStr
    """

    @log_call
    @xmlrpc_timeout
    def PointTableUpdateLua(self, point_table_name, lua_file_name):
        try:

            rtn = self.robot.PointTableSwitch(point_table_name)  # Switch point table
            if rtn != 0:
                if rtn == RobotError.ERR_POINTTABLE_NOTFOUND:
                    error_str = "PointTable not Found!"
                else:
                    error_str = "PointTable not Found!"
                return rtn, error_str

            time.sleep(0.3)  # Add delay to ensure that the backend does receive the switched point table name after the switch

            result = self.robot.PointTableUpdateLua(lua_file_name)
            error_str = result[1]
            if not error_str:
                error_str = "fail to update lua, please inspect pointtable"
            return result[0], error_str

        except Exception as e:
            return RobotError.ERR_RPC_ERROR, ""

    """   
    @brief  Download the file
    @param  [in] fileType File Type    0-luadocument
    @param  [in] fileName File name    “test.lua”
    @param  [in] saveFilePath Save file path    “C：//test/”
    @return Error code success-0  fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def __FileDownLoad(self, fileType, fileName, saveFilePath):
        if not os.path.exists(saveFilePath):
            return RobotError.ERR_SAVE_FILE_PATH_NOT_FOUND
        rtn = self.robot.FileDownload(fileType, fileName)
        if rtn == -1:
            return RobotError.ERR_POINTTABLE_NOTFOUND
        elif rtn != 0:
            return rtn
        port = 20011
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.settimeout(2)
        try:
            client.connect((self.ip_address, port))
        except Exception as e:
            client.close()
            return RobotError.ERR_OTHER
        total_buffer = bytearray(1024 * 1024 * 50)  # 50Mb
        total_size = 0
        recv_md5 = ""
        recv_size = 0
        find_head_flag = False
        while True:
            buffer = client.recv(1024)
            length = len(buffer)
            if length < 1:
                return RobotError.ERR_OTHER
            total_buffer[total_size:total_size + len(buffer)] = buffer
            total_size += len(buffer)
            if not find_head_flag and total_size > 4 and total_buffer[:4].decode('utf-8') == "/f/b":
                find_head_flag = True
            # After finding the file header，Extract file size andMD5Verification code。The file size information is in the total data5Arrive12Bytes，MD5The verification code information is located in the total data13Arrive44Bytes。
            if find_head_flag and total_size > 12 + 32:
                recv_size = int(total_buffer[4:12].decode('utf-8'))
                recv_md5 = total_buffer[12:44].decode('utf-8')
            # Receive the entire file and jump out of the loop
            if find_head_flag and total_size == recv_size:
                break
        if total_size == 0:
            return RobotError.ERR_OTHER
        file_buffer = total_buffer[12 + 32:total_size - 4]
        with open(os.path.join(saveFilePath, fileName), 'wb') as file_writer:
            file_writer.write(file_buffer[:total_size - 16 - 32])
        check_md5 = calculate_file_md5(saveFilePath + fileName)
        if check_md5 == recv_md5:
            client.send("SUCCESS".encode('utf-8'))
            return 0
        else:
            client.send("FAIL".encode('utf-8'))
            os.remove(os.path.join(saveFilePath, fileName))
            return RobotError.ERR_OTHER

    """   
    @brief  Upload file
    @param  [in] fileType File Type    0-luadocument
    @param  [in] filePathFull path name of uploaded file    C://test/test.lua     
    @return Error code success-0  fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def __FileUpLoad(self, fileType, filePath):

        if not os.path.exists(filePath):
            return RobotError.ERR_POINTTABLE_NOTFOUND

        MAX_UPLOAD_FILE_SIZE = 500 * 1024 * 1024;  # The maximum uploaded file is500Mb
        file_info = os.path.getsize(filePath)
        total_size = file_info + 46 + 4
        if total_size > MAX_UPLOAD_FILE_SIZE:
            print("Files larger than 500 MB are not supported!")
            return -1
        file_name = os.path.basename(filePath)
        rtn = self.robot.FileUpload(fileType, file_name)
        if rtn != 0:
            return rtn

        port = 20010

        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.settimeout(20)

        try:
            client.connect((self.ip_address, port))
        except Exception as e:
            client.close()
            return RobotError.ERR_OTHER
        client.settimeout(20)

        send_md5 = calculate_file_md5(filePath)
        head_data = f"/f/b{total_size:10d}{send_md5}"
        num = client.send(head_data.encode('utf-8'))

        if num < 1:
            return RobotError.ERR_OTHER

        with open(filePath, "rb") as f:
            while True:
                data = f.read(2 * 1024 * 1024)
                if not data:  # If read to the end of the file
                    end_data = "/b/f"
                    num = client.send(end_data.encode('utf-8'))  # Send file transfer flag
                    if num < 1:
                        return RobotError.ERR_OTHER
                    break  # Break out of the loop
                num = client.send(data)  # Pass the read data throughsocketSend the connection to the client
                if num < 1:
                    return RobotError.ERR_OTHER
        time.sleep(0.5)
        result_buf = client.recv(1024)
        if result_buf[:7].decode('utf-8') == "SUCCESS":
            return RobotError.ERR_SUCCESS
        else:
            return RobotError.ERR_OTHER

    """   
    @brief  Delete files
    @param  [in] fileType File Type    0-luadocument
    @param  [in] fileName File name    “test.lua”
    @return Error code success-0  fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def __FileDelete(self, fileType, fileName):
        rtn = self.robot.FileDelete(fileType, fileName)
        return rtn

    """   
    @brief  downloadLuadocument
    @param  [in] fileName To be downloadedluafile name“test.lua”
    @param  [in] savePath Save file local path“D://Down/”
    @return Error code success-0  fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def LuaDownLoad(self, fileName, savePath):
        error = self.__FileDownLoad(0, fileName, savePath)
        return error

    """   
    @brief  UploadLuadocument
    @param  [in] filePathFull path name of uploaded file   C://test/test.lua  
    @return Error code success-0  fail-Error code
    """

    def LuaUpload(self, filePath):
        error = self.__FileUpLoad(0, filePath)
        if error == 0:
            file_name = os.path.basename(filePath)
            _error = self.robot.LuaUpLoadUpdate(file_name)
            tmp_error = _error[0]
            if tmp_error == 0:
                return tmp_error
            else:
                return tmp_error, _error[1]
        else:
            return error

    """   
    @brief  deleteLuadocument
    @param  [in] fileName To be deletedluafile name“test.lua”
    @return Error code success-0  fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def LuaDelete(self, fileName):
        error = self.__FileDelete(0, fileName)
        return error

    """   
    @brief  Get all the current onesluaFile name
    @return Error code success-0  fail-Error code
    @return Return value（The call returns successfully） lua_num luaNumber of files
    @return Return value（The call returns successfully） luaNames luaFile name list
    """

    @log_call
    @xmlrpc_timeout
    def GetLuaList(self):
        _error = self.robot.GetLuaList()
        size = len(_error)
        error = _error[0]
        if _error[0] == 0:
            lua_num = _error[1]
            lua_name = _error[2].split(';')
            return error, lua_num, lua_name
        else:
            return error

    """   
    @brief  set up485Extended axis parameters
    @param  [in] Required parameters int servoId Servo driveID，scope[1-16],Corresponding slaveID 
    @param  [in] Required parameters int servoCompany Servo drive manufacturer，1-Dynatech
    @param  [in] Required parameters int servoModel Servo drive model，1-FD100-750C
    @param  [in] Required parameters int servoSoftVersion Servo drive software version，1-V1.0
    @param  [in] Required parameters int servoResolution Encoder resolution
    @param  [in] Required parameters float axisMechTransRatio Mechanical transmission ratio  
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def AuxServoSetParam(self, servoId, servoCompany, servoModel, servoSoftVersion, servoResolution,
                         axisMechTransRatio):
        servoId = int(servoId)
        servoCompany = int(servoCompany)
        servoModel = int(servoModel)
        servoSoftVersion = int(servoSoftVersion)
        servoResolution = int(servoResolution)
        axisMechTransRatio = float(axisMechTransRatio)
        error = self.robot.AuxServoSetParam(servoId, servoCompany, servoModel, servoSoftVersion, servoResolution,
                                            axisMechTransRatio)
        return error

    """   
    @brief  Get485Extended axis configuration parameters
    @param  [in] Required parameters int servoId Servo driveID，scope[1-16],Corresponding slaveID 
    @return Error code success- 0, fail-Error code
    @return Return value（The call returns successfully） int servoCompany Servo drive manufacturer，1-Dynatech
    @return Return value（The call returns successfully） servoModel Servo drive model，1-FD100-750C 
    @return Return value（The call returns successfully） servoSoftVersion Servo drive software version，1-V1.0
    @return Return value（The call returns successfully） int servoResolution Encoder resolution
    @return Return value（The call returns successfully） float axisMechTransRatio Mechanical transmission ratio
    """

    @log_call
    @xmlrpc_timeout
    def AuxServoGetParam(self, servoId):
        servoId = int(servoId)
        _error = self.robot.AuxServoGetParam(servoId)
        error = _error[0]
        if _error[0] == 0:
            return error, _error[1], _error[2], _error[3], _error[4], _error[5]
        else:
            return error

    """   
    @brief  set up485Extended axis enable/Go to enable
    @param  [in] Required parameters int servoId Servo driveID，scope[1-16],Corresponding slaveID 
    @param  [in] Required parameters int status Enable status，0-Go to enable， 1-Enable
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def AuxServoEnable(self, servoId, status):
        servoId = int(servoId)
        status = int(status)
        error = self.robot.AuxServoEnable(servoId, status)
        return error

    """   
    @brief  set up485Extended axis control mode
    @param  [in] Required parameters int servoId Servo driveID，scope[1-16],Corresponding slaveID 
    @param  [in] Required parameters mode Control mode，0-Position mode，1-Speed ​​mode
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def AuxServoSetControlMode(self, servoId, mode):
        servoId = int(servoId)
        mode = int(mode)
        error = self.robot.AuxServoSetControlMode(servoId, mode)
        return error

    """   
    @brief  set up485Extended axis target position(Position mode)
    @param  [in] Required parameters int servoId Servo driveID，scope[1-16],Corresponding slaveID 
    @param  [in] Required parameters float pos Target location，mmor°
    @param  [in] Required parameters float speed Target speed，mm/sor°/s
    @param  [in] Required parameters acc Acceleration percentage[0-100] 
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def AuxServoSetTargetPos(self, servoId, pos, speed,acc):
        servoId = int(servoId)
        pos = float(pos)
        speed = float(speed)
        acc = float(acc)
        error = self.robot.AuxServoSetTargetPos(servoId, pos, speed,acc)
        return error

    """   
    @brief  set up485Extended axis target speed(Speed ​​mode)
    @param  [in] Required parameters int servoId Servo driveID，scope[1-16],Corresponding slaveID 
    @param  [in] Required parameters float speed Target speed，mm/sor°/s
    @param  [in] Required parameters acc Acceleration percentage[0-100] 
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def AuxServoSetTargetSpeed(self, servoId, speed,acc):
        servoId = int(servoId)
        speed = float(speed)
        acc = float(acc)
        error = self.robot.AuxServoSetTargetSpeed(servoId, speed,acc)
        return error

    """   
    @brief  set up485Extended shaft target torque(Torque mode)
    @param  [in] Required parameters int servoId Servo driveID，scope[1-16],Corresponding slaveID 
    @param  [in] Required parameters float torque Target torque，Nm
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def AuxServoSetTargetTorque(self, servoId, torque):
        servoId = int(servoId)
        torque = float(torque)
        error = self.robot.AuxServoSetTargetTorque(servoId, torque)
        return error

    """   
    @brief  set up485Expanded axis back to zero
    @param  [in] Required parameters int servoId Servo driveID，scope[1-16],Corresponding slaveID 
    @param  [in] Required parameters int mode Return to zero mode，1-The current position returns to zero；2-Negative limit back to zero；3-Positive limit back to zero
    @param  [in] Required parameters float searchVel Return to zero speed，mm/sor°/s
    @param  [in] Required parameters float latchVel Hoop speed，mm/sor°/s
    @param  [in] Required parameters acc Acceleration percentage[0-100] 
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def AuxServoHoming(self, servoId, mode, searchVel, latchVel,acc):
        servoId = int(servoId)
        mode = int(mode)
        searchVel = float(searchVel)
        latchVel = float(latchVel)
        acc = float(acc)
        error = self.robot.AuxServoHoming(servoId, mode, searchVel, latchVel,acc)
        return error

    """   
    @brief  Clear485Expanded axis error message
    @param  [in] Required parameters int servoId Servo driveID，scope[1-16],Corresponding slaveID 
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def AuxServoClearError(self, servoId):
        servoId = int(servoId)
        error = self.robot.AuxServoClearError(servoId)
        return error

    """   
    @brief  Get485Extended axis servo state
    @param  [in] Required parameters int servoId Servo driveID，scope[1-16],Corresponding slaveID 
    @return Error code success- 0, fail-Error code
    @return Return value（The call returns successfully） servoErrCode Servo drive fault code
    @return Return value（The call returns successfully） servoState Servo Drive Status bit0:0-Not enabled；1-Enable;  bit1:0-Not exercising；1-In exercise;bit2:0-Positive limit not triggered，1-Positive limit trigger；bit3:0-Negative limit not triggered，1-Negative limit trigger；   bit4 0-Not positioned completed；1-Positioning is completed；  bit5：0-No return to zero；1-Return to zero completed
    @return Return value（The call returns successfully） servoPos Current position of the servo mmor°
    @return Return value（The call returns successfully） servoSpeed Current servo speed mm/sor°/s
    @return Return value（The call returns successfully） servoTorque Current servo torqueNm
    """

    @log_call
    @xmlrpc_timeout
    def AuxServoGetStatus(self, servoId):
        servoId = int(servoId)
        _error = self.robot.AuxServoGetStatus(servoId)
        error = _error[0]
        if _error[0] == 0:
            return error, _error[1], _error[2], _error[3], _error[4], _error[5]
        else:
            return error

    """   
    @brief  Set status feedback485Extended axis data shaft number
    @param  [in] Required parameters int servoId Servo driveID，scope[1-16],Corresponding slaveID 
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def AuxServosetStatusID(self, servoId):
        servoId = int(servoId)
        error = self.robot.AuxServoSetStatusID(servoId)
        return error

    """   
    @brief  Set up robot peripheral protocol
    @param  [in] Required parameters int protocol Robot peripheral protocol number 4096-Expanding shaft control card；4097-ModbusSlave；4098-ModbusMaster
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def SetExDevProtocol(self, protocol):
        protocol = int(protocol)
        error = self.robot.SetExDevProtocol(protocol)
        return error

    """   
    @brief  Obtain the robot peripheral protocol
    @return Error code success- 0, fail-Error code
    @return Return value（The call returns successfully） int protocol Robot peripheral protocol number 4096-Expanding shaft control card；4097-ModbusSlave；4098-ModbusMaster
    """

    @log_call
    @xmlrpc_timeout
    def GetExDevProtocol(self):
        _error = self.robot.GetExDevProtocol()
        error = _error[0]
        if _error[0] == 0:
            return error, _error[1]
        else:
            return error

    """   
    @brief  Set the robot acceleration
    @param [in]Required parameters acc Robot acceleration percentage
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def SetOaccScale(self, acc):
        acc = float(acc)
        error = self.robot.SetOaccScale(acc)
        return error

    """   
    @brief  Control boxAOFlying shot begins
    @param [in]Required parameters AONum Control boxAOserial number
    @param [in]Default parameters maxTCPSpeed maximumTCPSpeed ​​value[1-5000mm/s]，default1000
    @param [in]Default parameters maxAOPercent maximumTCPThe corresponding speed valueAOpercentage，default100%
    @param [in]Required parameters zeroZoneCmp Dead-band compensation valueAOpercentage，Plastic surgery，Default is20%，scope[0-100]
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def MoveAOStart(self, AONum, maxTCPSpeed=1000, maxAOPercent=100, zeroZoneCmp=20):
        AONum = int(AONum)
        maxTCPSpeed = int(maxTCPSpeed)
        maxAOPercent = int(maxAOPercent)
        zeroZoneCmp = int(zeroZoneCmp)
        error = self.robot.MoveAOStart(AONum, maxTCPSpeed, maxAOPercent, zeroZoneCmp)
        return error

    """   
    @brief  Control boxAOFlying shot stop
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def MoveAOStop(self):
        error = self.robot.MoveAOStop()
        return error

    """   
    @brief  EndAOFlying shot begins
    @param [in]Required parameters AONum EndAOserial number
    @param [in]Required parameters maxTCPSpeed maximumTCPSpeed ​​value[1-5000mm/s]，default1000
    @param [in]Required parameters maxAOPercent maximumTCPThe corresponding speed valueAOpercentage，default100%
    @param [in]Required parameters zeroZoneCmp Dead-band compensation valueAOpercentage，Plastic surgery，Default is20%，scope[0-100]
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def MoveToolAOStart(self, AONum, maxTCPSpeed=1000, maxAOPercent=100, zeroZoneCmp=20):
        AONum = int(AONum)
        maxTCPSpeed = int(maxTCPSpeed)
        maxAOPercent = int(maxAOPercent)
        zeroZoneCmp = int(zeroZoneCmp)
        error = self.robot.MoveToolAOStart(AONum, maxTCPSpeed, maxAOPercent, zeroZoneCmp)
        return error

    """   
    @brief  EndAOFlying shot stop
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def MoveToolAOStop(self):
        error = self.robot.MoveToolAOStop()
        return error

    """   
    @brief  UDPExtended axis communication parameter configuration
    @param [in]Required parameters ip PLC IPaddress
    @param [in]Required parameters port	Port number
    @param [in]Required parameters period Communication cycle(ms，Not open yet)
    @param [in]Required parameters lossPkgTime	Packet loss detection time(ms)
    @param [in]Required parameters lossPkgNum	Number of packet loss
    @param [in]Required parameters disconnectTime	Communication disconnection confirmation time
    @param [in]Required parameters reconnectEnable	Automatic reconnection is enabled for communication disconnection 0-不使能 1-Enable
    @param [in]Required parameters reconnectPeriod	Re-continuous week interval(ms)
    @param [in]Required parameters reconnectNum	重连次数
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def ExtDevSetUDPComParam(self, ip, port, period, lossPkgTime, lossPkgNum, disconnectTime,
                             reconnectEnable, reconnectPeriod, reconnectNum):
        ip = str(ip)
        port = int(port)
        period = int(period)
        period = 2  # Not open yet，Must be2
        lossPkgTime = int(lossPkgTime)
        lossPkgNum = int(lossPkgNum)
        disconnectTime = int(disconnectTime)
        reconnectEnable = int(reconnectEnable)
        reconnectPeriod = int(reconnectPeriod)
        reconnectNum = int(reconnectNum)

        error = self.robot.ExtDevSetUDPComParam(ip, port, period, lossPkgTime, lossPkgNum, disconnectTime,
                                                reconnectEnable, reconnectPeriod, reconnectNum)
        return error

    """   
    @brief  GetUDPExtended axis communication parameters
    @return Error code success- 0, fail-Error code
    @return Return value（The call returns successfully）ip PLC IPaddress
    @return Return value（The call returns successfully）port	Port number
    @return Return value（The call returns successfully）period Communication cycle(ms，Not open yet)
    @return Return value（The call returns successfully） lossPkgTime	Packet loss detection time(ms)
    @return Return value（The call returns successfully） lossPkgNum	Number of packet loss
    @return Return value（The call returns successfully） disconnectTime	Communication disconnection confirmation time
    @return Return value（The call returns successfully） reconnectEnable	Automatic reconnection is enabled for communication disconnection 0-Not enabled 1-Enable
    @return Return value（The call returns successfully） reconnectPeriod	Re-continuous week interval(ms)
    @return Return value（The call returns successfully） reconnectNum	Number of reconnections
    """

    @log_call
    @xmlrpc_timeout
    def ExtDevGetUDPComParam(self):
        _error = self.robot.ExtDevGetUDPComParam()
        if _error[0] == 0:
            return _error[0], [_error[1], _error[2], _error[3], _error[4], _error[5], _error[6], _error[7], _error[8],
                               _error[9]]
        else:
            return _error[0]

    """   
    @brief  loadUDPCommunication
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def ExtDevLoadUDPDriver(self):
        error = self.robot.ExtDevLoadUDPDriver()
        return error

    """   
    @brief  uninstallUDPCommunication
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def ExtDevUnloadUDPDriver(self):
        error = self.robot.ExtDevUnloadUDPDriver()
        return error

    """   
    @brief  UDPResuming connection after abnormal disconnection of the expansion axis
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def ExtDevUDPClientComReset(self):
        error = self.robot.ExtDevUDPClientComReset()
        return error

    """   
    @brief  UDPCommunication is turned off after the expansion axis is abnormally disconnected
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def ExtDevUDPClientComClose(self):
        error = self.robot.ExtDevUDPClientComClose()
        return error

    """   
    @brief  Set the relative expansion axis position of the expansion robot
    @param [in]Required parameters installType 0-The robot is mounted on the external shaft，1-The robot is installed outside the external axis
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def SetRobotPosToAxis(self, installType):
        installType = int(installType)
        error = self.robot.SetRobotPosToAxis(installType)
        return error

    """   
    @brief  Set up the extension axis systemDHParameter configuration
    @param [in]Required parameters axisConfig External shaft configuration，0-Single-degree of freedom linear slide rail，1-Two degrees of freedomLType transformer，2-Three degrees of freedom，3-Four degrees of freedom，4-Single degree of freedom converter
    @param [in]Required parameters  axisDHd1 External axisDHparameterd1 mm
    @param [in]Required parameters  axisDHd2 External axisDHparameterd2 mm
    @param [in]Required parameters  axisDHd3 External axisDHparameterd3 mm
    @param [in]Required parameters  axisDHd4 External axisDHparameterd4 mm
    @param [in]Required parameters  axisDHa1 External axisDHparametera1 mm
    @param [in]Required parameters  axisDHa2 External axisDHparametera2 mm
    @param [in]Required parameters  axisDHa3 External axisDHparametera3 mm
    @param [in]Required parameters  axisDHa4 External axisDHparametera4 mm
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def SetAxisDHParaConfig(self, axisConfig, axisDHd1, axisDHd2, axisDHd3, axisDHd4, axisDHa1, axisDHa2, axisDHa3,
                            axisDHa4):
        axisConfig = int(axisConfig)
        axisDHd1 = float(axisDHd1)
        axisDHd2 = float(axisDHd2)
        axisDHd3 = float(axisDHd3)
        axisDHd4 = float(axisDHd4)
        axisDHa1 = float(axisDHa1)
        axisDHa2 = float(axisDHa2)
        axisDHa3 = float(axisDHa3)
        axisDHa4 = float(axisDHa4)
        error = self.robot.SetAxisDHParaConfig(axisConfig, axisDHd1, axisDHd2, axisDHd3, axisDHd4, axisDHa1, axisDHa2,
                                               axisDHa3, axisDHa4)
        return error

    """   
    @brief  UDPExtended axis parameter configuration
    @param [in]Required parameters axisId Axle number[1-4]
    @param [in]Required parameters axisType Extended axis type 0-Pan；1-Rotate
    @param [in]Required parameters axisDirection Expanding axis direction 0-Positive；1-Reverse
    @param [in]Required parameters axisMax Maximum position of the expansion axis mm
    @param [in]Required parameters axisMin Minimum position of the expansion axis mm
    @param [in]Required parameters axisVel speedmm/s
    @param [in]Required parameters axisAcc Accelerationmm/s2
    @param [in]Required parameters axisLead Guidemm
    @param [in]Required parameters encResolution Encoder resolution
    @param [in]Required parameters axisOffect Weld start point expansion shaft offset
    @param [in]Required parameters axisCompany Driver manufacturer 1-Hechuan；2-Huichuan；3-Matsushita
    @param [in]Required parameters axisModel Drive model 1-Hechuan-SV-XD3EA040L-E，2-Hechuan-SV-X2EA150A-A，1-Huichuan-SV620PT5R4I，1-Matsushita-MADLN15SG，2-Matsushita-MSDLN25SG，3-Matsushita-MCDLN35SG
    @param [in]Required parameters axisEncType Encoder type  0-Increment；1-Absolute value
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def ExtAxisParamConfig(self, axisId, axisType, axisDirection, axisMax, axisMin, axisVel, axisAcc, axisLead,
                           encResolution, axisOffect, axisCompany, axisModel, axisEncType):
        axisId = int(axisId)
        axisType = int(axisType)
        axisDirection = int(axisDirection)
        axisMax = float(axisMax)
        axisMin = float(axisMin)
        axisVel = float(axisVel)
        axisAcc = float(axisAcc)
        axisLead = float(axisLead)
        encResolution = int(encResolution)
        axisOffect = float(axisOffect)
        axisCompany = int(axisCompany)
        axisModel = int(axisModel)
        axisEncType = int(axisEncType)
        error = self.robot.ExtAxisParamConfig(axisId, axisType, axisDirection, axisMax, axisMin, axisVel, axisAcc,
                                              axisLead, encResolution, axisOffect, axisCompany, axisModel, axisEncType)
        return error

    """   
    @brief  Get the configuration information of the extended shaft driver
    @param [in]Required parameters axisId Axle number[1-4]
    @return Error code success- 0, fail-Error code
    @return Return value（The call returns successfully） axisCompany Driver manufacturer 1-Hechuan；2-Huichuan；3-Matsushita
    @return Return value（The call returns successfully） axisModel Drive model 1-Hechuan-SV-XD3EA040L-E，2-Hechuan-SV-X2EA150A-A，1-Huichuan-SV620PT5R4I，1-Matsushita-MADLN15SG，2-Matsushita-MSDLN25SG，3-Matsushita-MCDLN35SG
    @return Return value（The call returns successfully） axisEncType Encoder type  0-Increment；1-Absolute value
    """

    @log_call
    @xmlrpc_timeout
    def GetExAxisDriverConfig(self, axisId):
        axisId = int(axisId)
        error = self.robot.GetExAxisDriverConfig(axisId)
        if error[0] == 0:
            return error[0], [error[1], error[2], error[3]]
        else:
            return error

    """   
    @brief  Set the reference point of the extended axis coordinate system-Four-point method
    @param [in]Required parameters pointNum Point number[1-4]
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def ExtAxisSetRefPoint(self, pointNum):
        pointNum = int(pointNum)
        error = self.robot.ExtAxisSetRefPoint(pointNum)
        return error

    """   
    @brief  Calculate the extended axis coordinate system-Four-point method
    @return Error code success- 0, fail-Error code
    @return Return value（The call returns successfully） coord Coordinate system value[x,y,z,rx,ry,rz]
    """

    @log_call
    @xmlrpc_timeout
    def ExtAxisComputeECoordSys(self):
        error = self.robot.ExtAxisComputeECoordSys()
        return error

    """   
    @brief  Set the position of the calibration reference point under the coordinate system of the terminal transformer
    @param [in]Required parameters pos Pose value[x,y,z,rx,ry,rz]
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def SetRefPointInExAxisEnd(self, pos):
        pos = list(map(float, pos))
        error = self.robot.SetRefPointInExAxisEnd(pos[0], pos[1], pos[2], pos[3], pos[4], pos[5])
        return error

    """   
    @brief  Setting of reference point of the coordinate system of the displacement machine
    @param [in]Required parameters pointNum Point number[1-4]
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def PositionorSetRefPoint(self, pointNum):
        pointNum = int(pointNum)
        error = self.robot.PositionorSetRefPoint(pointNum)
        return error

    """   
    @brief  Calculation of coordinate system of positioning machine-Four-point method
    @return Error code success- 0, fail-Error code
    @return Return value（The call returns successfully） coord Coordinate system value[x,y,z,rx,ry,rz]
    """

    @log_call
    @xmlrpc_timeout
    def PositionorComputeECoordSys(self):
        _error = self.robot.PositionorComputeECoordSys()
        error = _error[0]
        if error == 0:
            return error, [_error[1], _error[2], _error[3], _error[4], _error[5], _error[6]]
        else:
            return error

    """   
    @brief  Apply extended axis coordinate system
    @param [in]Required parameters axisCoordNum Coordinate system number
    @param [in]Required parameters toolNum Tool number
    @param [in]Required parameters coord Coordinate system value[x,y,z,rx,ry,rz]
    @param [in]Required parameters calibFlag Calibration mark 0-no，1-yes    
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def ExtAxisActiveECoordSys(self, axisCoordNum, toolNum, coord, calibFlag):
        axisCoordNum = int(axisCoordNum)
        toolNum = int(toolNum)
        coord = list(map(float, coord))
        calibFlag = int(calibFlag)
        error = self.robot.ExtAxisActiveECoordSys(axisCoordNum, toolNum, coord[0], coord[1], coord[2], coord[3],
                                                  coord[4], coord[5], calibFlag)
        return error

    """   
    @brief  UDPExtended axis enable
    @param [in]Required parameters axisID Axle number[1-4]
    @param [in]Required parameters status 0-Go to enable；1-Enable
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def ExtAxisServoOn(self, axisID, status):
        axisID = int(axisID)
        status = int(status)
        error = self.robot.ExtAxisServoOn(axisID, status)
        return error

    """   
    @brief  UDPExpanded axis back to zero
    @param [in]Required parameters axisID Axle number[1-4]
    @param [in]Required parameters mode Return to zero method 0The current position returns to zero，1Negative limit back to zero，2-Positive limit back to zero
    @param [in]Required parameters searchVel Zero speed(mm/s)
    @param [in]Required parameters latchVel Zero hoop speed(mm/s)
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def ExtAxisSetHoming(self, axisID, mode, searchVel, latchVel):
        axisID = int(axisID)
        mode = int(mode)
        searchVel = float(searchVel)
        latchVel = float(latchVel)
        error = self.robot.ExtAxisSetHoming(axisID, mode, searchVel, latchVel)
        return error

    """   
    @brief  UDPExpanding axis jog starts
    @param [in]Required parameters axisID Axle number[1-4]
    @param [in]Required parameters direction Rotation direction 0-Reverse；1-Positive
    @param [in]Required parameters vel speed(mm/s)
    @param [in]Required parameters acc (Acceleration mm/s2)
    @param [in]Required parameters maxDistance Maximum jog distance
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def ExtAxisStartJog(self, axisID, direction, vel, acc, maxDistance):
        if self.GetSafetyCode() != 0:
            return self.GetSafetyCode()
        axisID = int(axisID)
        direction = int(direction)
        vel = float(vel)
        acc = float(acc)
        maxDistance = float(maxDistance)
        error = self.robot.ExtAxisStartJog(6, axisID, direction, vel, acc, maxDistance)
        return error

    """   
    @brief  UDPExtended axis jog stop
    @param [in]Required parameters axisID Axle number[1-4]
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def ExtAxisStopJog(self, axisID):
        axisID = int(axisID)
        error =self.send_message("/f/bIII19III240III14IIIStopExtAxisJogIII/b/f")
        # error = self.robot.ExtAxisStartJog(7, axisID, 0, 0.0, 0.0, 0.0)
        return error

    """   
    @brief  Setting up extensionsDO
    @param [in]Required parameters DONum DOserial number
    @param [in]Required parameters bOpen switch True-open,False-close
    @param [in]Required parameters smooth Is it smooth or not True -yes, False -no
    @param [in]Required parameters block Whether it is blocked True -yes, False -no
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def SetAuxDO(self, DONum, bOpen, smooth, block):
        DONum = int(DONum)
        bOpen = bool(bOpen)
        smooth = bool(smooth)
        block = bool(block)
        open_flag = 1 if bOpen else 0
        smooth_flag = 1 if smooth else 0
        no_block_flag = 1 if block else 0
        print("open_flag",open_flag)
        print("smooth_flag", smooth_flag)
        print("no_block_flag", no_block_flag)
        error = self.robot.SetAuxDO(DONum, open_flag, smooth_flag, no_block_flag)
        return error

    """   
    @brief  Setting up extensionsAO
    @param [in]Required parameters AONum AOserial number 
    @param [in]Required parameters value Analog value[0-4095]
    @param [in]Required parameters block Whether it is blocked True-yes,False-no
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def SetAuxAO(self, AONum, value, block):
        AONum = int(AONum)
        value = float(value)
        block = bool(block)
        no_block_flag = 0 if block else 1
        value =value*40.96
        error = self.robot.SetAuxAO(AONum, value, no_block_flag)
        return error

    """   
    @brief  Setting up extensionsDIInput filter time
    @param [in]Required parameters filterTime Filtering time(ms)
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def SetAuxDIFilterTime(self, filterTime):
        filterTime = int(filterTime)
        error = self.robot.SetAuxDIFilterTime(filterTime)
        return error

    """   
    @brief  Setting up extensionsAIInput filter time
    @param [in]Required parameters AINum AIserial number
    @param [in]Required parameters filterTime Filtering time(ms)
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def SetAuxAIFilterTime(self, AINum,filterTime):
        AINum = int(AINum)
        filterTime = int(filterTime)
        error = self.robot.SetAuxAIFilterTime(AINum,filterTime)
        return error

    """   
    @brief  Wait for expansionDIenter
    @param [in]Required parameters DINum DIserial number
    @param [in]Required parameters bOpen switch True-open,False-close
    @param [in]Required parameters time Maximum waiting time(ms)
    @param [in]Required parameters errorAlarm Whether to continue exercise True-yes,False-no
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def WaitAuxDI(self, DINum, bOpen, time, errorAlarm):
        DINum = int(DINum)
        bOpen = bool(bOpen)
        open_flag = 0 if bOpen else 1
        time = int(time)
        errorAlarm = bool(errorAlarm)
        errorAlarm_flag = 0 if errorAlarm else 1
        error = self.robot.WaitAuxDI(DINum, open_flag, time, errorAlarm_flag)
        return error

    """   
    @brief  Wait for expansionAIenter
    @param [in]Required parameters AINum AIserial number
    @param [in]Required parameters sign 0-Greater than；1-Less than
    @param [in]Required parameters value AIvalue
    @param [in]Required parameters time Maximum waiting time(ms)
    @param [in]Required parameters errorAlarm Whether to continue exercise True-yes,False-no
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def WaitAuxAI(self, AINum, sign, value, time, errorAlarm):
        AINum = int(AINum)
        sign = int(sign)
        value = int(value)
        time = int(time)
        errorAlarm = bool(errorAlarm)
        errorAlarm_flag = 0 if errorAlarm else 1
        error = self.robot.WaitAuxAI(AINum, sign, value, time, errorAlarm_flag)
        return error

    """   
    @brief  Get extensionDIvalue
    @param [in]Required parameters DINum DIserial number
    @param [in]Required parameters isNoBlock Whether it is blocked True-block false-Non-blocking
    @return Error code success- 0, fail-Error code
    @return Return value（The call returns successfully） isOpen 0-close；1-open
    """

    @log_call
    @xmlrpc_timeout
    def GetAuxDI(self, DINum, isNoBlock):
        DINum = int(DINum)
        isNoBlock = bool(isNoBlock)
        isNoBlock_flag = 0 if isNoBlock else 1
        error = self.robot.GetAuxDI(DINum, isNoBlock_flag)
        if error[0] == 0:
            return error[0], error[1]
        else:
            return error

    """   
    @brief  Get extensionAIvalue
    @param [in]Required parameters AINum AIserial number
    @param [in]Required parameters isNoBlock Whether it is blocked True-block False-Non-blocking
    @return Error code success- 0, fail-Error code
    @return Return value（The call returns successfully） value Enter a value
    """

    @log_call
    @xmlrpc_timeout
    def GetAuxAI(self, AINum, isNoBlock):
        AINum = int(AINum)
        isNoBlock = bool(isNoBlock)
        isNoBlock_flag = 0 if isNoBlock else 1
        error = self.robot.GetAuxAI(AINum, isNoBlock_flag)
        if error[0] == 0:
            return error[0], error[1]
        else:
            return error

    """   
    @brief  UDPExtended axis motion
    @param [in]Required parameters pos Target location axis 1 Location ~ axis 4 Location[exaxis[0],exaxis[1],exaxis[2],exaxis[3]]
    @param [in]Required parameters ovl Speed ​​percentage
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def ExtAxisMove(self, pos, ovl):
        if self.GetSafetyCode() != 0:
            return self.GetSafetyCode()
        pos = list(map(float, pos))
        ovl = float(ovl)
        error = self.robot.ExtAxisMoveJ(0, pos[0], pos[1], pos[2], pos[3], ovl)
        return error

    """   
    @brief  UDPSynchronous motion of the expansion axis and robot joint motion
    @param  [in] Required parameters joint_pos: Target joint position，unit [°]
    @param  [in] Required parameters desc_pos: Target Cartesian pose，unit [mm][°]
    @param  [in] Required parameters tool: Tool number，[0~14]
    @param  [in] Required parameters user: Workpiece number，[0~14]
    @param  [in] Required parameters exaxis_pos: External axis 1 Location ~ External axis 4 Location 
    @param  [in] Default parameters vel：Speed ​​percentage，[0~100] default20.0
    @param  [in] Default parameters acc：Acceleration percentage，[0~100] Not open yet,default0.0 
    @param  [in] Default parameters ovl: Speed ​​scaling factor，[0~100] default100.0    
    @param  [in] Default parameters blendT:[-1.0]-Exercise in place (block)，[0~500.0]-Smoothing time (Non-blocking)，unit [ms] default-1.0
    @param  [in] Default parameters offset_flag:[0]-No offset，[1]-Workpieces/Offset under base coordinate system，[2]-Offset under tool coordinate system default 0
    @param  [in] Default parameters offset_pos: Pose offset，unit [mm][°] default[0.0,0.0,0.0,0.0,0.0,0.0]
    @return Error code success-0  fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def ExtAxisSyncMoveJ(self, joint_pos, desc_pos, tool, user, exaxis_pos, vel=20.0, acc=0.0, ovl=100.0,
                         blendT=-1.0, offset_flag=0, offset_pos=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]):
        if self.GetSafetyCode() != 0:
            return self.GetSafetyCode()
        joint_pos = list(map(float, joint_pos))
        tool = int(tool)
        user = int(user)
        desc_pos = list(map(float, desc_pos))
        vel = float(vel)
        acc = float(acc)
        ovl = float(ovl)
        exaxis_pos = list(map(float, exaxis_pos))
        blendT = float(blendT)
        offset_flag = int(offset_flag)
        offset_pos = list(map(float, offset_pos))
        if (desc_pos[0] == 0.0) and (desc_pos[1] == 0.0) and (desc_pos[2] == 0.0) and (desc_pos[3] == 0.0) and (
                desc_pos[4] == 0.0) and (desc_pos[5] == 0.0):  # If no parameters are entered, call positive kinematics to solve
            ret = self.robot.GetForwardKin(joint_pos)  # Solve the kinematics
            if ret[0] == 0:
                desc_pos = [ret[1], ret[2], ret[3], ret[4], ret[5], ret[6]]
            else:
                error = ret[0]
                return error
        error = self.robot.ExtAxisMoveJ(1, exaxis_pos[0], exaxis_pos[1], exaxis_pos[2], exaxis_pos[3], ovl)
        if error != 0:
            return error
        error = self.robot.MoveJ(joint_pos, desc_pos, tool, user, vel, acc, ovl, exaxis_pos, blendT, offset_flag,
                                 offset_pos)
        return error

    """   
    @brief  UDPSynchronous motion of the expansion axis and the linear motion of the robot
    @param  [in] Required parameters joint_pos: Target joint position，unit [°] 
    @param  [in] Required parameters desc_pos: Target Cartesian pose，unit [mm][°]
    @param  [in] Required parameters tool: Tool number，[0~14]
    @param  [in] Required parameters user: Workpiece number，[0~14]
    @param  [in] Required parameters exaxis_pos: External axis 1 Location ~ External axis 4 Location 
    @param  [in] Default parameters vel：Speed ​​percentage，[0~100] default20.0
    @param  [in] Default parameters acc：Acceleration percentage，[0~100] Not open yet default0.0
    @param  [in] Default parameters ovl: Speed ​​scaling factor，[0~100] default100.0
    @param  [in] Default parameters blendR:[-1.0]-Exercise in place (block)，[0~1000]-Smooth radius (Non-blocking)，unit [mm] default-1.0    
    @param  [in] Default parameters search:[0]-No welding wire position search，[1]-Wire positioning
    @param  [in] Default parameters offset_flag:[0]-No offset，[1]-Workpieces/Offset under base coordinate system，[2]-Offset under tool coordinate system default 0
    @param  [in] Default parameters offset_pos: Pose offset，unit [mm][°] default[0.0,0.0,0.0,0.0,0.0,0.0]
    @return Error code success-0  fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def ExtAxisSyncMoveL(self, joint_pos, desc_pos, tool, user, exaxis_pos, vel=20.0, acc=0.0, ovl=100.0,
                         blendR=-1.0, search=0, offset_flag=0, offset_pos=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]):
        if self.GetSafetyCode() != 0:
            return self.GetSafetyCode()
        desc_pos = list(map(float, desc_pos))
        tool = int(tool)
        user = int(user)
        joint_pos = list(map(float, joint_pos))
        vel = float(vel)
        acc = float(acc)
        ovl = float(ovl)
        blendR = float(blendR)
        exaxis_pos = list(map(float, exaxis_pos))
        search = int(search)
        offset_flag = int(offset_flag)
        offset_pos = list(map(float, offset_pos))
        if ((joint_pos[0] == 0.0) and (joint_pos[1] == 0.0) and (joint_pos[2] == 0.0) and (joint_pos[3] == 0.0)
                and (joint_pos[4] == 0.0) and (joint_pos[5] == 0.0)):  # If no parameters are entered, inverse kinematics is called to solve the problem.
            ret = self.robot.GetInverseKin(0, desc_pos, -1)  # Resolving inverse kinematics
            if ret[0] == 0:
                joint_pos = [ret[1], ret[2], ret[3], ret[4], ret[5], ret[6]]
            else:
                error = ret[0]
                return error
        error = self.robot.ExtAxisMoveJ(1, exaxis_pos[0], exaxis_pos[1], exaxis_pos[2], exaxis_pos[3], ovl)
        if error != 0:
            return error
        error = self.robot.MoveL(joint_pos, desc_pos, tool, user, vel, acc, ovl, blendR, exaxis_pos, search,
                                 offset_flag, offset_pos)
        return error

    """   
    @brief  UDPSynchronous motion of the expansion axis and the robot arc motion
    @param  [in] Required parameters joint_pos_p: Path point joint position，unit [°] 
    @param  [in] Required parameters desc_pos_p: Cartesian pose，unit [mm][°]
    @param  [in] Required parameters tool_p: Pathpoint tool number，[0~14]
    @param  [in] Required parameters user_p: Path point workpiece number，[0~14]
    @param  [in] Required parameters exaxis_pos_p: Path point external axis 1 Location ~ External axis 4 Location default[0.0,0.0,0.0,0.0]
    @param  [in] Required parameters joint_pos_t: Target joint position，unit [°] 
    @param  [in] Required parameters desc_pos_t: Cartesian pose of the target point，unit [mm][°]
    @param  [in] Required parameters tool_t: Tool number，[0~14]
    @param  [in] Required parameters user_t: Workpiece number，[0~14]
    @param  [in] Required parameters exaxis_pos_t: Target point external axis 1 Location ~ External axis 4 Location default[0.0,0.0,0.0,0.0]    
    @param  [in] Default parameters vel_p: Path point velocity percentage，[0~100] default20.0
    @param  [in] Default parameters acc_p: Path point acceleration percentage，[0~100] Not open yet,default0.0    
    @param  [in] Default parameters offset_flag_p: Is the path point offset?[0]-No offset，[1]-Workpieces/Offset under base coordinate system，[2]-Offset under tool coordinate system default 0
    @param  [in] Default parameters offset_pos_p: Path Point Position Offset，unit [mm][°] default[0.0,0.0,0.0,0.0,0.0,0.0]
    @param  [in] Default parameters vel_t: Target point speed percentage，[0~100] default20.0
    @param  [in] Default parameters acc_t: Target point acceleration percentage，[0~100] Not open yet default0.0
    @param  [in] Default parameters offset_flag_t: Is the target point offset?[0]-No offset，[1]-Workpieces/Offset under base coordinate system，[2]-Offset under tool coordinate system default 0
    @param  [in] Default parameters offset_pos_t: Target Point Position Offset，unit [mm][°] default[0.0,0.0,0.0,0.0,0.0,0.0]
    @param  [in] Default parameters ovl: Speed ​​scaling factor，[0~100] default100.0
    @param  [in] Default parameters blendR:[-1.0]-Exercise in place (block)，[0~1000]-Smooth radius (Non-blocking)，unit [mm] default-1.0
    @return Error code success-0  fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def ExtAxisSyncMoveC(self, joint_pos_p, desc_pos_p, tool_p, user_p, exaxis_pos_p, joint_pos_t, desc_pos_t, tool_t,
                         user_t, exaxis_pos_t,
                         vel_p=20.0, acc_p=100.0, offset_flag_p=0,
                         offset_pos_p=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                         vel_t=20.0, acc_t=100.0, offset_flag_t=0,
                         offset_pos_t=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                         ovl=100.0, blendR=-1.0):
        if self.GetSafetyCode() != 0:
            return self.GetSafetyCode()
        desc_pos_p = list(map(float, desc_pos_p))
        tool_p = float(int(tool_p))
        user_p = float(int(user_p))
        joint_pos_p = list(map(float, joint_pos_p))
        vel_p = float(vel_p)
        acc_p = float(acc_p)
        exaxis_pos_p = list(map(float, exaxis_pos_p))
        offset_flag_p = int(offset_flag_p)
        offset_pos_p = list(map(float, offset_pos_p))

        desc_pos_t = list(map(float, desc_pos_t))
        tool_t = float(int(tool_t))
        user_t = float(int(user_t))
        joint_pos_t = list(map(float, joint_pos_t))
        vel_t = float(vel_t)
        acc_t = float(acc_t)
        exaxis_pos_t = list(map(float, exaxis_pos_t))
        offset_flag_t = int(offset_flag_t)
        offset_pos_t = list(map(float, offset_pos_t))

        ovl = float(ovl)
        blendR = float(blendR)

        if ((joint_pos_p[0] == 0.0) and (joint_pos_p[1] == 0.0) and (joint_pos_p[2] == 0.0) and (joint_pos_p[3] == 0.0)
                and (joint_pos_p[4] == 0.0) and (joint_pos_p[5] == 0.0)):  # If no parameters are entered, inverse kinematics is called to solve the problem.
            retp = self.robot.GetInverseKin(0, desc_pos_p, -1)  # Resolving inverse kinematics
            if retp[0] == 0:
                joint_pos_p = [retp[1], retp[2], retp[3], retp[4], retp[5], retp[6]]
            else:
                error = retp[0]
                return error

        if ((joint_pos_t[0] == 0.0) and (joint_pos_t[1] == 0.0) and (joint_pos_t[2] == 0.0) and (joint_pos_t[3] == 0.0)
                and (joint_pos_t[4] == 0.0) and (joint_pos_t[5] == 0.0)):  # If no parameters are entered, inverse kinematics is called to solve the problem.
            rett = self.robot.GetInverseKin(0, desc_pos_t, -1)  # Resolving inverse kinematics
            if rett[0] == 0:
                joint_pos_t = [rett[1], rett[2], rett[3], rett[4], rett[5], rett[6]]
            else:
                error = rett[0]
                return error
        error = self.robot.ExtAxisMoveJ(1, exaxis_pos_t[0], exaxis_pos_t[1], exaxis_pos_t[2], exaxis_pos_t[3], ovl)
        if error != 0:
            return error
        error = self.robot.MoveC(joint_pos_p, desc_pos_p, [tool_p, user_p, vel_p, acc_p], exaxis_pos_p, offset_flag_p,
                                 offset_pos_p, joint_pos_t, desc_pos_t, [tool_t, user_t, vel_t, acc_t], exaxis_pos_t,
                                 offset_flag_t, offset_pos_t, ovl, blendR)
        return error

    """   
    @brief  The welding wire positioning starts
    @param [in]Required parameters refPos  1-Benchmark point 2-Contact point
    @param [in]Required parameters searchVel   Location search speed %
    @param [in]Required parameters searchDis  Location search distance mm
    @param [in]Required parameters autoBackFlag Automatic return flag，0-Not automatically；-automatic
    @param [in]Required parameters autoBackVel  Automatic return speed %
    @param [in]Required parameters autoBackDis  Automatic return distance mm
    @param [in]Required parameters offectFlag  1-Position search with offset；2-Teaching point to find a location
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def WireSearchStart(self, refPos,searchVel,searchDis,autoBackFlag,autoBackVel,autoBackDis,offectFlag):
        refPos = int(refPos)
        searchVel = float(searchVel)
        searchDis = int(searchDis)
        autoBackFlag = int(autoBackFlag)
        autoBackVel = float(autoBackVel)
        autoBackDis = int(autoBackDis)
        offectFlag = int(offectFlag)
        error = self.robot.WireSearchStart(refPos,searchVel,searchDis,autoBackFlag,autoBackVel,autoBackDis,offectFlag)
        return error

    """   
    @brief  The wire position search ends
    @param [in]Required parameters refPos  1-Benchmark point 2-Contact point
    @param [in]Required parameters searchVel   Location search speed %
    @param [in]Required parameters searchDis  Location search distance mm
    @param [in]Required parameters autoBackFlag Automatic return flag，0-Not automatically；-automatic
    @param [in]Required parameters autoBackVel  Automatic return speed %
    @param [in]Required parameters autoBackDis  Automatic return distance mm
    @param [in]Required parameters offectFlag  1-Position search with offset；2-Teaching point to find a location
    @return Error code success- 0, fail-Error code
    """
    @log_call
    @xmlrpc_timeout
    def WireSearchEnd(self, refPos,searchVel,searchDis,autoBackFlag,autoBackVel,autoBackDis,offectFlag):
        refPos = int(refPos)
        searchVel = float(searchVel)
        searchDis = int(searchDis)
        autoBackFlag = int(autoBackFlag)
        autoBackVel = float(autoBackVel)
        autoBackDis = int(autoBackDis)
        offectFlag = int(offectFlag)
        error = self.robot.WireSearchEnd(refPos,searchVel,searchDis,autoBackFlag,autoBackVel,autoBackDis,offectFlag)
        return error

    """   
    @brief  Calculate the welding wire positioning offset
    @param  [in]Required parameters seamType  Weld type
    @param  [in]Required parameters method   Calculation method
    @param  [in]Required parameters varNameRef Benchmark point1-6，“#”Indicates no point variables
    @param  [in]Required parameters varNameRes Contact point1-6，“#”Indicates no point variables
    @return Error code success- 0, fail-Error code
    @return Return value（The call returns successfully） offectFlag 0-The offset is directly superimposed to the instruction point；1-The offset requires coordinate transformation of the instruction point
    @return Return value（The call returns successfully） offect Offset position[x, y, z, a, b, c]
    """
    @log_call
    @xmlrpc_timeout
    def GetWireSearchOffset(self, seamType, method,varNameRef,varNameRes):
        seamType = int(seamType)
        method = int(method)
        if(len(varNameRes)!=6):
            return 4
        if(len(varNameRes)!=6):
            return 4
        varNameRef = list(map(str, varNameRef))
        varNameRes = list(map(str, varNameRes))

        _error = self.robot.GetWireSearchOffset(seamType, method, varNameRef[0], varNameRef[1], varNameRef[2], varNameRef[3], varNameRef[4], varNameRef[5],
                                               varNameRes[0], varNameRes[1], varNameRes[2], varNameRes[3], varNameRes[4], varNameRes[5])
        error = _error[0]
        if error == 0:
            return error, _error[1], [_error[2], _error[3], _error[4], _error[5], _error[6], _error[7]]
        else:
            return error

    """   
    @brief  Wait for the welding wire to find the positioning
    @return Error code success- 0, fail-Error code
    """
    @log_call
    @xmlrpc_timeout
    def WireSearchWait(self,varname):
        varname=str(varname)
        error = self.robot.WireSearchWait(varname)
        return error

    """   
    @brief  Wire positioning contact point writing to database
    @param  [in]Required parameters varName  Contact point name “RES0” ~ “RES99”
    @param  [in]Required parameters pos  Contact point data[x, y, x, a, b, c]
    @return Error code success- 0, fail-Error code
    """
    @log_call
    @xmlrpc_timeout
    def SetPointToDatabase(self,varName,pos):
        varName = str(varName)
        pos = list(map(float,pos))

        error = self.robot.SetPointToDatabase(varName,pos)
        return error

    """   
    @brief  Arc tracking control
    @param  [in]Required parameters flag switch，0-close；1-open
    @param  [in]Required parameters delayTime Lag time，unitms
    @param  [in]Required parameters isLeftRight Left-left deviation compensation 0-closure，1-Open
    @param  [in]Required parameters klr Left and left adjustment coefficient(Sensitivity)
    @param  [in]Required parameters tStartLr Start compensation timecyc
    @param  [in]Required parameters stepMaxLr Maximum compensation per time mm
    @param  [in]Required parameters sumMaxLr The maximum compensation amount of left and right mm
    @param  [in]Required parameters isUpLow Up and down deviation compensation 0-closure，1-Open
    @param  [in]Required parameters kud Up and down adjustment coefficient(Sensitivity)
    @param  [in]Required parameters tStartUd Compensation time up and downcyc
    @param  [in]Required parameters stepMaxUd Maximum compensation amount for each up and down mm
    @param  [in]Required parameters sumMaxUd Maximum compensation amount up and down
    @param  [in]Required parameters axisSelect Up and down coordinate system selection，0-swing；1-tool；2-Pedestal
    @param  [in]Required parameters referenceType Up and down reference current setting method，0-feedback；1-constant
    @param  [in]Required parameters referSampleStartUd Up and down reference current sampling start counting(feedback)，cyc
    @param  [in]Required parameters referSampleCountUd Up and down reference current sampling cycle count(feedback)，cyc
    @param  [in]Required parameters referenceCurrent Up and down reference currentmA
    @return Error code success- 0, fail-Error code
    """
    @log_call
    @xmlrpc_timeout
    def ArcWeldTraceControl(self,flag,delaytime, isLeftRight, klr, tStartLr, stepMaxLr, sumMaxLr, isUpLow, kud, tStartUd, stepMaxUd,
                            sumMaxUd, axisSelect, referenceType, referSampleStartUd, referSampleCountUd, referenceCurrent):
        flag = int(flag)
        delaytime = float(delaytime)
        isLeftRight = int(isLeftRight)
        klr = float(klr)
        tStartLr = float(tStartLr)
        stepMaxLr = float(stepMaxLr)
        sumMaxLr = float(sumMaxLr)
        isUpLow = int(isUpLow)
        kud = float(kud)
        tStartUd = float(tStartUd)
        stepMaxUd = float(stepMaxUd)
        sumMaxUd = float(sumMaxUd)
        axisSelect = int(axisSelect)
        referenceType = int(referenceType)
        referSampleStartUd = float(referSampleStartUd)
        referSampleCountUd = float(referSampleCountUd)
        referenceCurrent = float(referenceCurrent)

        error = self.robot.ArcWeldTraceControl(flag,delaytime, isLeftRight, [klr, tStartLr, stepMaxLr, sumMaxLr], isUpLow, [kud, tStartUd, stepMaxUd,
                            sumMaxUd], axisSelect, referenceType, referSampleStartUd, referSampleCountUd, referenceCurrent)
        return error

    """   
    @brief  Arc trackingAIPassband selection
    @param  [in]Required parameters channel Arc trackingAIPassband selection,[0-3]
    @return Error code success- 0, fail-Error code
    """
    @log_call
    @xmlrpc_timeout
    def ArcWeldTraceExtAIChannelConfig(self,channel):
        channel = int(channel)
        error = self.robot.ArcWeldTraceExtAIChannelConfig(channel)
        return error

    """   
    @brief  Force sensor assisted drag
    @param  [in]Required parameters status Control status，0-closure；1-Open
    @param  [in]Required parameters asaptiveFlag Adaptive turn-on sign，0-closure；1-Open
    @param  [in]Required parameters interfereDragFlag Interference zone drag sign，0-closure；1-Open
    @param  [in]Required parameters M=[m1,m2,m3,m4,m5,m6] Coefficient of inertia 
    @param  [in]Required parameters B=[b1,b2,b3,b4,b5,b6] Damping coefficient
    @param  [in]Required parameters K=[k1,k2,k3,k4,k5,k6] 刚度系数
    @param  [in]Required parameters F=[f1,f2,f3,f4,f5,f6] Drag the six-dimensional force threshold
    @param  [in]Required parameters Fmax Maximum drag force limit
    @param  [in]Required parameters Vmax Maximum joint speed limit
    @return Error code success- 0, fail-Error code
    """
    @log_call
    @xmlrpc_timeout
    def EndForceDragControl(self, status, asaptiveFlag, interfereDragFlag, M, B, K, F, Fmax, Vmax):
        status = int(status)
        asaptiveFlag = int(asaptiveFlag)
        interfereDragFlag = int(interfereDragFlag)
        M = list(map(float,M))
        B = list(map(float,B))
        K = list(map(float,K))
        F = list(map(float,F))
        Fmax = float(Fmax)
        Vmax = float(Vmax)
        error = self.robot.EndForceDragControl(status, asaptiveFlag, interfereDragFlag, M, B, K, F, Fmax, Vmax)
        return error

    """   
    @brief  The force sensor will automatically turn on after the error is cleared
    @param  [in]Required parameters status Control status，0-closure；1-Open
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def SetForceSensorDragAutoFlag(self, status):
        status = int(status)
        error = self.robot.SetForceSensorDragAutoFlag(status)
        return error

    """   
    @brief  Setting up a mixed drag switch and parameters for six-dimensional force and joint impedance
    @param  [in]Required parameters status Control status，0-closure；1-Open
    @param  [in]Required parameters impedanceFlag Impedance on flag，0-closure；1-Open
    @param  [in]Required parameters lamdeDain Drag Gain
    @param  [in]Required parameters KGain Stiffness gain
    @param  [in]Required parameters BGain Damping Gain
    @param  [in]Required parameters dragMaxTcpVel Drag the end maximum linear speed limit
    @param  [in]Required parameters dragMaxTcpOriVel Drag the end maximum angular velocity limit
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def ForceAndJointImpedanceStartStop(self,status, impedanceFlag, lamdeDain, KGain, BGain,dragMaxTcpVel,dragMaxTcpOriVel):
        status = int(status)
        impedanceFlag = int(impedanceFlag)
        if((len(lamdeDain)!=6)or(len(KGain)!=6)or(len(BGain)!=6)):
            return 4
        lamdeDain = list(map(float,lamdeDain))
        KGain = list(map(float,KGain))
        BGain = list(map(float,BGain))
        dragMaxTcpVel = float(dragMaxTcpVel)
        dragMaxTcpOriVel = float(dragMaxTcpOriVel)
        error = self.robot.ForceAndJointImpedanceStartStop(status, impedanceFlag, lamdeDain, KGain, BGain,dragMaxTcpVel,dragMaxTcpOriVel)
        return error


    """   
    @brief  Get the state of dragging switch for force sensor
    @return Error code success- 0, fail-Error code
    @return Return value（The call returns successfully） dragState Force sensor assisted drag control status，0-closure；1-Open
    @return Return value（The call returns successfully） sixDimensionalDragState Six-dimensional force assisted drag control status，0-closure；1-Open
    """

    @log_call
    @xmlrpc_timeout
    def GetForceAndTorqueDragState(self):
        _error = self.robot.GetForceAndTorqueDragState()
        error = _error[0]
        if error == 0:
            return error, _error[1], _error[2]
        else:
            return error

    """   
    @brief  Set the load weight under the force sensor
    @param  [in]Required parameters weight Load weight kg
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def SetForceSensorPayload(self,weight):
        weight = float(weight)
        error = self.robot.SetForceSensorPayload(weight)
        return error

    """   
    @brief  Set the load weight under the force sensor
    @param  [in]Required parameters x Load centroidx mm 
    @param  [in]Required parameters y Load centroidy mm 
    @param  [in]Required parameters z Load centroidz mm 
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def SetForceSensorPayloadCog(self,x,y,z):
        x = float(x)
        y = float(y)
        z = float(z)
        error = self.robot.SetForceSensorPayloadCog(x,y,z)
        return error

    """   
    @brief  Obtain the load weight under the force sensor
    @return Error code success- 0, fail-Error code
    @return Return value（The call returns successfully） weight Load weight kg
    """

    @log_call
    @xmlrpc_timeout
    def GetForceSensorPayload(self):
        _error = self.robot.GetForceSensorPayload()
        error = _error[0]
        if error == 0:
            return error, _error[1]
        else:
            return error


    """   
    @brief  Obtain the load centroid under the force sensor
    @return Error code success- 0, fail-Error code
    @return Return value（The call returns successfully） x Load centroidx mm 
    @return Return value（The call returns successfully） y Load centroidy mm 
    @return Return value（The call returns successfully） z Load centroidz mm 
    """

    @log_call
    @xmlrpc_timeout
    def GetForceSensorPayloadCog(self):
        _error = self.robot.GetForceSensorPayloadCog()
        error = _error[0]
        if error == 0:
            return error, _error[1], _error[2], _error[3]
        else:
            return error

    """   
    @brief  Automatic zero-resolution of force sensor
    @return Error code success- 0, fail-Error code
    @return Return value（The call returns successfully） weight Sensor quality kg 
    @return Return value（The call returns successfully） pos=[x,y,z] Sensor centroid mm
    """
    @log_call
    @xmlrpc_timeout
    def ForceSensorAutoComputeLoad(self):
        rtn = self.ForceSensorSetSaveDataFlag(1)
        if rtn!=0:
            return rtn
        error =self.GetActualJointPosDegree()
        start_joint = error[1]
        error = self.GetActualJointPosDegree()
        if error[0]==0:
            joint =error[1]
            if joint[2]<0:
                joint[3] = joint[3] + 90
            else:
                joint[3] = joint[3] - 90
            rtn = self.MoveJ(joint,0,0,vel=10)
            if rtn!=0:
                return rtn
        else:
            return error

        rtn = self.ForceSensorSetSaveDataFlag(2)
        if rtn!=0:
            return rtn

        error = self.GetActualJointPosDegree()
        if error[0] == 0:
            joint = error[1]
            if joint[5] < 0:
                joint[5] = joint[5] + 90
            else:
                joint[5] = joint[5] - 90
            rtn = self.MoveJ(joint, 0, 0,vel=10)
            if rtn != 0:
                return rtn
        else:
            return error

        rtn = self.ForceSensorSetSaveDataFlag(3)
        if rtn!=0:
            return rtn

        _error = self.robot.ForceSensorComputeLoad()
        error = _error[0]
        self.MoveJ(start_joint,0,0,vel=10)
        if error == 0:
            return error, _error[1],[_error[2],_error[3],_error[4]]
        else:
            return error

    """   
    @brief  Automatic zero-recall data recording for sensors
    @param  [in]Required parameters recordCount Record the number of data 1-3
    @return Error code success- 0, fail-Error code
    """
    @log_call
    @xmlrpc_timeout
    def ForceSensorSetSaveDataFlag(self,recordCount):
        error = self.robot.ForceSensorSetSaveDataFlag(recordCount)
        return error

    """   
    @brief  Automatic zero-calibration calculation for sensors
    @return Error code success- 0, fail-Error code    
    @return Return value（The call returns successfully） weight Sensor quality kg 
    @return Return value（The call returns successfully） pos=[x,y,z] Sensor centroid mm
    """

    @log_call
    @xmlrpc_timeout
    def ForceSensorComputeLoad(self):
        _error = self.robot.ForceSensorComputeLoad()
        error = _error[0]
        if error == 0:
            return error, _error[1],[_error[2],_error[3],_error[4]]
        else:
            return error

    """   
    @brief  End sensor configuration
    @param  [in]Required parameters idCompany Manufacturer，18-JUNKONG；25-HUIDE
    @param  [in]Required parameters idDevice type，0-JUNKONG/RYR6T.V1.0
    @param  [in]Required parameters idSoftware Software version，0-J1.0/HuiDe1.0(Not open yet)
    @param  [in]Required parameters idBus Mounting location，1-End1Speaker；2-End2Speaker...8-End8Speaker(Not open yet)
    @return Error code success- 0, fail-Error code    
    """

    @log_call
    @xmlrpc_timeout
    def AxleSensorConfig(self,idCompany, idDevice, idSoftware, idBus):
        idCompany = int(idCompany)
        idDevice = int(idDevice)
        idSoftware = int(idSoftware)
        idBus = int(idBus)

        error = self.robot.AxleSensorConfig(idCompany, idDevice, idSoftware, idBus)
        return error

    """   
    @brief  Get end sensor configuration    
    @return Error code success- 0, fail-Error code  
    @return Return value（The call returns successfully） idCompany Manufacturer，18-JUNKONG；25-HUIDE
    @return Return value（The call returns successfully） idDevice type，0-JUNKONG/RYR6T.V1.0
    """

    @log_call
    @xmlrpc_timeout
    def AxleSensorConfigGet(self):
        _error = self.robot.AxleSensorConfigGet()
        error = _error[0]
        if error == 0:
            return error, _error[1], _error[2]
        else:
            return error

    """   
    @brief  End sensor activation
    @param  [in]Required parameters actFlag 0-Reset；1-activation
    @return Error code success- 0, fail-Error code    
    """

    @log_call
    @xmlrpc_timeout
    def AxleSensorActivate(self,actFlag):
        actFlag = int(actFlag)
        error = self.robot.AxleSensorActivate(actFlag)
        return error

    """   
    @brief  End sensor register writing
    @param  [in]Required parameters devAddr  Device address number 0-255
    @param  [in]Required parameters regHAddr Register address high8Bit
    @param  [in]Required parameters regLAddr Low register address8Bit
    @param  [in]Required parameters regNum  Number of registers 0-255
    @param  [in]Required parameters data1 Write register value1
    @param  [in]Required parameters data2 Write register value2
    @param  [in]Required parameters isNoBlock Whether it is blocked 0-block；1-Non-blocking
    @return Error code success- 0, fail-Error code    
    """

    @log_call
    @xmlrpc_timeout
    def AxleSensorRegWrite(self,devAddr, regHAddr, regLAddr, regNum, data1, data2, isNoBlock):
        devAddr = int(devAddr)
        regHAddr = int(regHAddr)
        regLAddr = int(regLAddr)
        regNum = int(regNum)
        data1 = int(data1)
        data2 = int(data2)
        isNoBlock = int(isNoBlock)
        error = self.robot.AxleSensorRegWrite(devAddr, regHAddr, regLAddr, regNum, data1, data2, isNoBlock)
        return error

    """   
    @brief  Set up the control boxDOstop/Whether the output is reset after pause
    @param  [in]Required parameters resetFlag  0-No reset；1-Reset
    @return Error code success- 0, fail-Error code    
    """

    @log_call
    @xmlrpc_timeout
    def SetOutputResetCtlBoxDO(self,resetFlag):
        resetFlag = int(resetFlag)
        error = self.robot.SetOutputResetCtlBoxDO(resetFlag)
        return error

    """   
    @brief  Set up the control boxAOstop/Whether the output is reset after pause
    @param  [in]Required parameters resetFlag  0-No reset；1-Reset
    @return Error code success- 0, fail-Error code    
    """

    @log_call
    @xmlrpc_timeout
    def SetOutputResetCtlBoxAO(self,resetFlag):
        resetFlag = int(resetFlag)
        error = self.robot.SetOutputResetCtlBoxAO(resetFlag)
        return error

    """   
    @brief  Setting the end toolDOstop/Whether the output is reset after pause
    @param  [in]Required parameters resetFlag  0-No reset；1-Reset
    @return Error code success- 0, fail-Error code    
    """

    @log_call
    @xmlrpc_timeout
    def SetOutputResetAxleDO(self,resetFlag):
        resetFlag = int(resetFlag)
        error = self.robot.SetOutputResetAxleDO(resetFlag)
        return error

    """   
    @brief  Setting the end toolAOstop/Whether the output is reset after pause
    @param  [in]Required parameters resetFlag  0-No reset；1-Reset
    @return Error code success- 0, fail-Error code    
    """

    @log_call
    @xmlrpc_timeout
    def SetOutputResetAxleAO(self,resetFlag):
        resetFlag = int(resetFlag)
        error = self.robot.SetOutputResetAxleAO(resetFlag)
        return error

    """   
    @brief  Setting up extensionsDOstop/Whether the output is reset after pause
    @param  [in]Required parameters resetFlag  0-No reset；1-Reset
    @return Error code success- 0, fail-Error code    
    """

    @log_call
    @xmlrpc_timeout
    def SetOutputResetExtDO(self,resetFlag):
        resetFlag = int(resetFlag)
        error = self.robot.SetOutputResetExtDO(resetFlag)
        return error

    """   
    @brief  Setting up extensionsAOstop/Whether the output is reset after pause
    @param  [in]Required parameters resetFlag  0-No reset；1-Reset
    @return Error code success- 0, fail-Error code    
    """

    @log_call
    @xmlrpc_timeout
    def SetOutputResetExtAO(self,resetFlag):
        resetFlag = int(resetFlag)
        error = self.robot.SetOutputResetExtAO(resetFlag)
        return error

    """   
    @brief  set upSmartToolstop/Whether the output is reset after pause
    @param  [in]Required parameters resetFlag  0-No reset；1-Reset
    @return Error code success- 0, fail-Error code    
    """

    @log_call
    @xmlrpc_timeout
    def SetOutputResetSmartToolDO(self,resetFlag):
        resetFlag = int(resetFlag)
        error = self.robot.SetOutputResetSmartToolDO(resetFlag)
        return error


    """   
    @brief  Simulation swing begins
    @param  [in]Required parameters weaveNum  Swing parameter number
    @return Error code success- 0, fail-Error code    
    """

    @log_call
    @xmlrpc_timeout
    def WeaveStartSim(self,weaveNum):
        weaveNum = int(weaveNum)
        error = self.robot.WeaveStartSim(weaveNum)
        return error

    """   
    @brief  Simulation swing ends
    @param  [in]Required parameters weaveNum  Swing parameter number
    @return Error code success- 0, fail-Error code    
    """

    @log_call
    @xmlrpc_timeout
    def WeaveEndSim(self,weaveNum):
        weaveNum = int(weaveNum)
        error = self.robot.WeaveEndSim(weaveNum)
        return error

    """   
    @brief  Start track detection warning(No exercise)
    @param  [in]Required parameters weaveNum  Swing parameter number
    @return Error code success- 0, fail-Error code    
    """

    @log_call
    @xmlrpc_timeout
    def WeaveInspectStart(self,weaveNum):
        weaveNum = int(weaveNum)
        error = self.robot.WeaveInspectStart(weaveNum)
        return error

    """   
    @brief  End track detection warning(No exercise)
    @param  [in]Required parameters weaveNum  Swing parameter number
    @return Error code success- 0, fail-Error code    
    """

    @log_call
    @xmlrpc_timeout
    def WeaveInspectEnd(self,weaveNum):
        weaveNum = int(weaveNum)
        error = self.robot.WeaveInspectEnd(weaveNum)
        return error



    """   
    @brief  Set welding process curve parameters
    @param  [in]Required parameters id Welding process number(1-99)
    @param  [in]Required parameters startCurrent Arc start current(A)
    @param  [in]Required parameters startVoltage Arc voltage(V)
    @param  [in]Required parameters startTime Arc start time(ms)
    @param  [in]Required parameters weldCurrent Welding current(A)
    @param  [in]Required parameters weldVoltage Welding voltage(V)
    @param  [in]Required parameters endCurrent Arc current(A)
    @param  [in]Required parameters endVoltage Arc voltage(V)
    @param  [in]Required parameters endTime Arc closing time(ms)
    @return Error code success- 0, fail-Error code    
    """

    @log_call
    @xmlrpc_timeout
    def WeldingSetProcessParam(self, id, startCurrent, startVoltage, startTime, weldCurrent, weldVoltage, endCurrent,
                               endVoltage, endTime):
        id = int(id)
        startCurrent = float(startCurrent)
        startVoltage = float(startVoltage)
        startTime = float(startTime)
        weldCurrent = float(weldCurrent)
        weldVoltage = float(weldVoltage)
        endCurrent = float(endCurrent)
        endVoltage = float(endVoltage)
        endTime = float(endTime)
        error = self.robot.WeldingSetProcessParam(id, startCurrent, startVoltage, startTime, weldCurrent, weldVoltage,
                                                  endCurrent, endVoltage, endTime)
        return error

    """   
    @brief  Obtain welding process curve parameters    
    @param  [in]Required parameters id Welding process number(1-99)
    @return Error code success- 0, fail-Error code  
    @return Return value（The call returns successfully） startCurrent Arc start current(A)
    @return Return value（The call returns successfully） startVoltage Arc voltage(V)
    @return Return value（The call returns successfully） startTime Arc start time(ms)
    @return Return value（The call returns successfully） weldCurrent Welding current(A)
    @return Return value（The call returns successfully） weldVoltage Welding voltage(V)
    @return Return value（The call returns successfully） endCurrent Arc current(A)
    @return Return value（The call returns successfully） endVoltage Arc voltage(V)
    @return Return value（The call returns successfully） endTime Arc closing time(ms)
    """

    @log_call
    @xmlrpc_timeout
    def WeldingGetProcessParam(self, id):
        id = int(id)
        _error = self.robot.WeldingGetProcessParam(id)
        error = _error[0]
        if error == 0:
            return error, _error[1], _error[2], _error[3], _error[4], _error[5], _error[6], _error[7], _error[8]
        else:
            return error

    """   
    @brief  ExtendedIO-Configure the gas detection signal of the welding machine
    @param  [in]Required parameters DONum  Gas detection signal extensionDOserial number
    @return Error code success- 0, fail-Error code    
    """

    @log_call
    @xmlrpc_timeout
    def SetAirControlExtDoNum(self,DONum):
        DONum = int(DONum)
        error = self.robot.SetAirControlExtDoNum(DONum)
        return error

    """   
    @brief  ExtendedIO-Configure the arc signal of the welding machine
    @param  [in]Required parameters DONum  Welder arc signal expansionDOserial number
    @return Error code success- 0, fail-Error code    
    """

    @log_call
    @xmlrpc_timeout
    def SetArcStartExtDoNum(self,DONum):
        DONum = int(DONum)
        error = self.robot.SetArcStartExtDoNum(DONum)
        return error

    """   
    @brief  ExtendedIO-Configure the welding machine reverse wire feed signal
    @param  [in]Required parameters DONum  Reverse wire feed signal extensionDOserial number
    @return Error code success- 0, fail-Error code    
    """

    @log_call
    @xmlrpc_timeout
    def SetWireReverseFeedExtDoNum(self,DONum):
        DONum = int(DONum)
        error = self.robot.SetWireReverseFeedExtDoNum(DONum)
        return error

    """   
    @brief  ExtendedIO-Configure the welder forward wire feed signal
    @param  [in]Required parameters DONum  Forward wire feeding signal extensionDOserial number
    @return Error code success- 0, fail-Error code    
    """

    @log_call
    @xmlrpc_timeout
    def SetWireForwardFeedExtDoNum(self,DONum):
        DONum = int(DONum)
        error = self.robot.SetWireForwardFeedExtDoNum(DONum)
        return error

    """   
    @brief  ExtendedIO-Configure the success signal of the welding machine
    @param  [in]Required parameters DINum  Arc success signal expansionDIserial number
    @return Error code success- 0, fail-Error code    
    """

    @log_call
    @xmlrpc_timeout
    def SetArcDoneExtDiNum(self,DINum):
        DINum = int(DINum)
        error = self.robot.SetArcDoneExtDiNum(DINum)
        return error

    """   
    @brief  ExtendedIO-Configure the welding machine to prepare the signal
    @param  [in]Required parameters DINum  Welding machine preparation signal expansionDIserial number
    @return Error code success- 0, fail-Error code    
    """

    @log_call
    @xmlrpc_timeout
    def SetWeldReadyExtDiNum(self,DINum):
        DINum = int(DINum)
        error = self.robot.SetWeldReadyExtDiNum(DINum)
        return error

    """   
    @brief  ExtendedIO-Configure welding interrupt recovery signal
    @param  [in]Required parameters reWeldDINum  Recover welding signal expansion after welding interruptionDIserial number
    @param  [in]Required parameters abortWeldDINum  Exit welding signal extension after welding interruptionDIserial number
    @return Error code success- 0, fail-Error code    
    """

    @log_call
    @xmlrpc_timeout
    def SetExtDIWeldBreakOffRecover(self,reWeldDINum, abortWeldDINum):
        reWeldDINum = int(reWeldDINum)
        abortWeldDINum = int(abortWeldDINum)
        error = self.robot.SetExtDIWeldBreakOffRecover(reWeldDINum, abortWeldDINum)
        return error



    """   
    @brief  Set up robot collision detection method
    @param  [in]Required parameters method Collision detection method：0-Current mode；1-Dual encoder；2-Current and dual encoder are turned on at the same time
    @return Error code success- 0, fail-Error code    
    """

    @log_call
    @xmlrpc_timeout
    def SetCollisionDetectionMethod(self,method):
        method = int(method)
        error = self.robot.SetCollisionDetectionMethod(method)
        return error

    """   
    @brief  Set the collision detection starts to close under static
    @param  [in]Required parameters status 0-closure；1-Open
    @return Error code success- 0, fail-Error code    
    """

    @log_call
    @xmlrpc_timeout
    def SetStaticCollisionOnOff(self,status):
        status = int(status)
        error = self.robot.SetStaticCollisionOnOff(status)
        return error

    """   
    @brief  Joint torque power detection
    @param  [in]Required parameters status 0-closure；1-Open
    @param  [in]Required parameters power Set maximum power(W)
    @return Error code success- 0, fail-Error code    
    """

    @log_call
    @xmlrpc_timeout
    def SetPowerLimit(self,status, power):
        status = int(status)
        power = float(power)
        error = self.robot.SetPowerLimit(status, power)
        return error

    """   
    @brief  Set up the robot 20004 Port feedback cycle
    @param  [in]Required parameters period robot 20004 Port feedback cycle(ms)
    @return Error code success- 0, fail-Error code    
    """

    @log_call
    @xmlrpc_timeout
    def SetRobotRealtimeStateSamplePeriod(self,period):
        period = int(period)
        error = self.robot.SetRobotRealtimeStateSamplePeriod(period)
        return error

    """   
    @brief  Get the robot 20004 Port feedback cycle
    @param  [in]NULL
    @return Error code success- 0, fail-Error code    
    @return Return value（The call returns successfully） period robot 20004 Port feedback cycle(ms)
    """

    @log_call
    @xmlrpc_timeout
    def GetRobotRealtimeStateSamplePeriod(self):
        _error = self.robot.GetRobotRealtimeStateSamplePeriod()
        error = _error[0]
        if error == 0:
            return error, _error[1]
        else:
            return error


    """   
    @brief  Get the current torque of the joint drive
    @param  [in] Required parameters NULL
    @return Error code success- 0, fail-Error code
    @return Return value（The call returns successfully）data=[j1,j2,j3,j4,j5,j6] Joint torque    [fx,fy,fz,tx,ty,tz]
    """
    @log_call
    @xmlrpc_timeout
    def GetJointDriverTorque(self):
        return 0,[self.robot_state_pkg.jointDriverTorque[0],self.robot_state_pkg.jointDriverTorque[1],self.robot_state_pkg.jointDriverTorque[2],
                  self.robot_state_pkg.jointDriverTorque[3],self.robot_state_pkg.jointDriverTorque[4],self.robot_state_pkg.jointDriverTorque[5]]


    """   
    @brief  Get the current temperature of the joint drive
    @param  [in] Required parameters NULL
    @return Error code success- 0, fail-Error code
    @return Return value（The call returns successfully）data=[t1,t2,t3,t4,t5,t6]
    """
    @log_call
    @xmlrpc_timeout
    def GetJointDriverTemperature (self):
        return 0,[self.robot_state_pkg.jointDriverTemperature [0],self.robot_state_pkg.jointDriverTemperature [1],self.robot_state_pkg.jointDriverTemperature[2],
                  self.robot_state_pkg.jointDriverTemperature [3],self.robot_state_pkg.jointDriverTemperature[4],self.robot_state_pkg.jointDriverTemperature[5]]



    """   
    @brief  Arc Tracking + Multi-layer and multi-channel compensation is opened
    @param  [in] Required parameters NULL
    @return Error code success- 0, fail-Error code    
    """

    @log_call
    @xmlrpc_timeout
    def ArcWeldTraceReplayStart(self):
        error = self.robot.ArcWeldTraceReplayStart()
        return error


    """   
    @brief  Arc Tracking + Multi-layer and multi-channel compensation closure
    @param  [in] Required parameters NULL
    @return Error code success- 0, fail-Error code    
    """

    @log_call
    @xmlrpc_timeout
    def ArcWeldTraceReplayEnd(self):
        error = self.robot.ArcWeldTraceReplayEnd()
        return error

    """   
    @brief  Offset coordinate changes-Multi-layer multi-pass welding
    @param  [in] pointo Cartesian pose in reference point
    @param  [in] pointX Benchmark pointXCartesian pose towards offset point
    @param  [in] pointZ Benchmark pointZCartesian pose towards offset point
    @param  [in] dx xDirection offset(mm)
    @param  [in] dz zDirection offset(mm)
    @param  [in] dry CircleyAxis offset(°)
    @return Error code success- 0, fail-Error code    
    @return Return value（The call returns successfully） offset Calculate the result offset
    """

    @log_call
    @xmlrpc_timeout
    def MultilayerOffsetTrsfToBase(self,pointo,pointX,pointZ,dx,dz,dry):
        pointo =list(map(float,pointo))
        pointX = list(map(float, pointX))
        pointZ = list(map(float, pointZ))
        dx = float(dx)
        dz = float(dz)
        dry = float(dry)
        _error = self.robot.MultilayerOffsetTrsfToBase(pointo[0],pointo[1],pointo[2],
                   pointX[0],pointX[1],pointX[2],pointZ[0],pointZ[1],pointZ[2],dx,dz,dry)
        error = _error[0]
        if error == 0:
            return error, [_error[1], _error[2], _error[3], _error[4], _error[5], _error[6]]
        else:
            return error


    """   
    @brief  Specified pose speed on
    @param  [in]Required parameters ratio Percentage of posture speed[0-300]
    @return Error code success- 0, fail-Error code    
    """
    @log_call
    @xmlrpc_timeout
    def AngularSpeedStart(self, ratio):
        ratio = int(ratio)
        error = self.robot.AngularSpeedStart(ratio)
        return error


    """   
    @brief  Specify pose speed off
    @return Error code success- 0, fail-Error code    
    """
    @log_call
    @xmlrpc_timeout
    def AngularSpeedEnd(self):
        error = self.robot.AngularSpeedEnd()
        return error


    """   
    @brief  Robot software upgrade
    @param  [in]Required parameters  filePath Software upgrade package full path
    @param  [in]Required parameters block Whether to block until the upgrade is completed true:block；false:Non-blocking
    @return Error code success- 0, fail-Error code    
    """
    @log_call
    @xmlrpc_timeout
    def SoftwareUpgrade(self,filePath, block):
        error = self.__FileUpLoad(1,filePath)

        print("__FileUpLoad", error)
        if 0==error:
            self.log_info("Software Upload success!")
            error =self.robot.SoftwareUpgrade()
            if 0!=error:
                return error
            if block:
                upgradeState = -1
                time.sleep(0.5)
                upgradeState = self.GetSoftwareUpgradeState()
                if upgradeState == 0:
                    self.log_error("software upgrade not start")
                    return -1
                while (upgradeState > 0 and upgradeState < 100):
                    time.sleep(0.5)
                    upgradeState = self.GetSoftwareUpgradeState()
                    # print("upgradeState",upgradeState,"%")
                if upgradeState == 100:
                    error = 0
                else:
                    error = upgradeState
            return error
        else:
            self.log_error("execute SoftwareUpgrade fail.")
            return error


    """   
    @brief  Get the robot software upgrade status
    @return Error code success- 0, fail-Error code   
    @return Return value（The call returns successfully） state Robot package upgrade status 0：Idle or uploading the upgrade package，1~100：Upgrade completion percentage，-1：Upgrading software failed，-2：Verification failed，-3：Version verification failed，-4：Decompression failed，-5：User configuration upgrade failed，-6：Peripheral configuration upgrade failed，-7：Expansion axis configuration upgrade failed，-8：Robot configuration upgrade failed，-9：DHParameter configuration upgrade failed
    """
    @log_call
    @xmlrpc_timeout
    def GetSoftwareUpgradeState(self):
        error = self.robot_state_pkg.softwareUpgradeState
        return error

    """   
    @brief  set up485Extended axis motion acceleration
    @param  [in]Required parameters  acc 485Extended axis motion acceleration
    @param  [in]Required parameters dec 485Extended axis motion deceleration
    @return Error code success- 0, fail-Error code    
    """
    @log_call
    @xmlrpc_timeout
    def AuxServoSetAcc(self,acc,dec):
        acc = float(acc)
        dec = float(dec)
        error = self.robot.AuxServoSetAcc(acc,dec)
        return error

    """   
    @brief  set up485Extended axis emergency stop acceleration speed
    @param  [in]Required parameters  acc 485Expanded axis emergency stop acceleration
    @param  [in]Required parameters dec 485Expanded shaft emergency stop and deceleration
    @return Error code success- 0, fail-Error code    
    """
    @log_call
    @xmlrpc_timeout
    def AuxServoSetEmergencyStopAcc(self,acc,dec):
        acc = float(acc)
        dec = float(dec)
        error = self.robot.AuxServoSetEmergencyStopAcc(acc,dec)
        return error


    """   
    @brief  Get485Extended axis emergency stop acceleration speed
    @return Error code success- 0, fail-Error code        
    @return Return value（The call returns successfully） acc 485Expanded axis emergency stop acceleration   
    @return Return value（The call returns successfully） dec 485Expanded shaft emergency stop and deceleration
    """
    @log_call
    @xmlrpc_timeout
    def AuxServoGetEmergencyStopAcc(self):
        error = self.robot.AuxServoGetEmergencyStopAcc()
        if error[0]==0:
            return error[0],error[1],error[2]
        else:
            return error

    """   
    @brief  Get485Extended axis motion acceleration
    @return Error code success- 0, fail-Error code        
    @return Return value（The call returns successfully） acc 485Extended axis motion acceleration 
    @return Return value（The call returns successfully） dec 485Extended axis motion deceleration
    """

    @log_call
    @xmlrpc_timeout
    def AuxServoGetAcc(self):
        error = self.robot.AuxServoGetAcc()
        if error[0] == 0:
            return error[0], error[1], error[2]
        else:
            return error

    """   
    @brief  Get end communication parameters
    @return Error code success- 0, fail-Error code        
    @return Return value（The call returns successfully） baudRate Baud rate：support 1-9600，2-14400，3-19200，4-38400，5-56000，6-67600，7-115200，8-128000；
    @return Return value（The call returns successfully） dataBit Data bits：Data bit support（8,9），Currently commonly used 8
    @return Return value（The call returns successfully） stopBit Stop position：1-1，2-0.5，3-2，4-1.5，Currently commonly used 1
    @return Return value（The call returns successfully） verify Check bit：0-None，1-Odd，2-Even,Currently commonly used 0；
    @return Return value（The call returns successfully） timeout Timeout：1~1000ms，This value needs to be combined with peripherals to set reasonable time parameters
    @return Return value（The call returns successfully） timeoutTimes  Timeouts：1~10，Mainly perform timeout retransmission，Reduce occasional abnormalities and improve user experience
    @return Return value（The call returns successfully） period Periodic instruction time interval：1~1000ms，Mainly used for the time interval of periodic instructions each time
    """

    @log_call
    @xmlrpc_timeout
    def GetAxleCommunicationParam(self):
        error = self.robot.GetAxleCommunicationParam()
        if error[0] == 0:
            return error[0], error[1], error[2], error[3], error[4], error[5], error[6], error[7]
        else:
            return error

    """   
    @brief  Set end communication parameters
    @param  [in]  baudRate Baud rate：support 1-9600，2-14400，3-19200，4-38400，5-56000，6-67600，7-115200，8-128000；
    @param  [in]  dataBit Data bits：Data bit support（8,9），Currently commonly used 8
    @param  [in]  stopBit Stop position：1-1，2-0.5，3-2，4-1.5，Currently commonly used 1
    @param  [in]  verify Check bit：0-None，1-Odd，2-Even,Currently commonly used 0；
    @param  [in]  timeout Timeout：1~1000ms，This value needs to be combined with peripherals to set reasonable time parameters
    @param  [in]  timeoutTimes  Timeouts：1~10，Mainly perform timeout retransmission，Reduce occasional abnormalities and improve user experience
    @param  [in]  period Periodic instruction time interval：1~1000ms，Mainly used for the time interval of periodic instructions each time
    @return Error code success- 0, fail-Error code        
    """
    @log_call
    @xmlrpc_timeout
    def SetAxleCommunicationParam(self,baudRate,dataBit,stopBit,verify,timeout,timeoutTimes,period):
        baudRate = int (baudRate)
        dataBit = int (dataBit)
        stopBit = int (stopBit)
        verify = int (verify)
        timeout = int (timeout)
        timeoutTimes = int (timeoutTimes)
        period = int(period)
        error = self.robot.SetAxleCommunicationParam(baudRate,dataBit,stopBit,verify,timeout,timeoutTimes,period)
        return error

    """   
    @brief  Set the end file transfer type
    @param  [in] type 1-MCUUpgrade files；2-LUAdocument
    @return Error code success- 0, fail-Error code        
    """
    @log_call
    @xmlrpc_timeout
    def SetAxleFileType(self,type):
        type=int(type)
        error = self.robot.SetAxleFileType(type)
        return error

    """   
    @brief  Setting the Enable EndLUAimplement
    @param  [in] enable 0-Not enabled；1-Enable
    @return Error code success- 0, fail-Error code        
    """
    @log_call
    @xmlrpc_timeout
    def SetAxleLuaEnable(self,enable):
        enable=int(enable)
        error = self.robot.SetAxleLuaEnable(enable)
        return error

    """   
    @brief  EndLUAFile exception error recovery
    @param  [in] status 0-No recovery；1-recover
    @return Error code success- 0, fail-Error code        
    """
    @log_call
    @xmlrpc_timeout
    def SetRecoverAxleLuaErr(self,enable):
        error = self.robot.SetRecoverAxleLuaErr(enable)
        return error

    """   
    @brief  Get the endLUAExecution enable status
    @return Error code success- 0, fail-Error code   
    @return Return value（The call returns successfully） enable 0-Not enabled；1-Enable
    """
    @log_call
    @xmlrpc_timeout
    def GetAxleLuaEnableStatus(self):
        error = self.robot.GetAxleLuaEnableStatus()
        if error[0] == 0:
            return error[0], error[1]
        else:
            return error

    """   
    @brief  Set the endLUAEnd device enable type
    @param  [in] forceSensorEnable Force sensor enabled status，0-Not enabled；1-Enable
    @param  [in] gripperEnable Claw enabled status，0-Not enabled；1-Enable
    @param  [in] IOEnable IODevice enabled status，0-Not enabled；1-Enable
    @return Error code success- 0, fail-Error code        
    """
    @log_call
    @xmlrpc_timeout
    def SetAxleLuaEnableDeviceType(self,forceSensorEnable,gripperEnable,IOEnable):
        forceSensorEnable = int(forceSensorEnable)
        gripperEnable = int(gripperEnable)
        IOEnable = int(IOEnable)
        error = self.robot.SetAxleLuaEnableDeviceType(forceSensorEnable,gripperEnable,IOEnable)
        return error

    """   
    @brief  Get the endLUAEnd device enable type
    @return Error code success- 0, fail-Error code   
    @return Return value（The call returns successfully） forceSensorEnable Force sensor enabled status，0-Not enabled；1-Enable
    @return Return value（The call returns successfully） gripperEnable Claw enabled status，0-Not enabled；1-Enable
    @return Return value（The call returns successfully） IOEnable IODevice enabled status，0-Not enabled；1-Enable
    """
    @log_call
    @xmlrpc_timeout
    def GetAxleLuaEnableDeviceType(self):
        error = self.robot.GetAxleLuaEnableDeviceType()
        if error[0] == 0:
            return error[0], error[1], error[2], error[3]
        else:
            return error

    """   
    @brief  Get the current configuration end device
    @return Error code success- 0, fail-Error code   
    @return Return value（The call returns successfully） forceSensorEnable[8] Force sensor enabled status，0-Not enabled；1-Enable
    @return Return value（The call returns successfully） gripperEnable[8] Claw enabled status，0-Not enabled；1-Enable
    @return Return value（The call returns successfully） IOEnable[8]  IODevice enabled status，0-Not enabled；1-Enable
    """
    @log_call
    @xmlrpc_timeout
    def GetAxleLuaEnableDevice(self):
        error = self.robot.GetAxleLuaEnableDevice()
        if error[0] == 0:
            par= error[1].split(',')
            if 24 != len(par):
                self.log_error("GetAxleLuaEnableDevice fail")
                return -1
            else:
                print(par)
                return (error[0], [par[0],par[1], par[2], par[3], par[4], par[5], par[6], par[7], par[8]],
                        [par[9], par[10], par[11], par[12], par[13], par[14], par[15], par[16]],
                        [par[17], par[18], par[19], par[20], par[21], par[22], par[23]])
        else:
            return error

    """   
    @brief  Set to enable the jaw motion control function
    @param  [in] id Claw equipment number
    @param  [in] func 0-Claw enable；1-Initialization of jaws；2-Location settings；3-Speed ​​settings；4-Torque setting；6-Read the jaw status；7-Read the initialization state；8-Read the fault code；9-Reading location；10-Reading speed；11-Read torque,12-15Reservation
    @return Error code success- 0, fail-Error code        
    """
    @log_call
    @xmlrpc_timeout
    def SetAxleLuaGripperFunc(self,id,func):
        id = int(id)
        func = list(map(int, func))
        error = self.robot.SetAxleLuaGripperFunc(id,func)
        # error = self.robot.SetAxleLuaGripperFunc(id,func)
        return error

    """   
    @brief  Get the function of enabling jaw motion control
    @param  [in] id Claw equipment number
    @return Error code success- 0, fail-Error code   
    @return Return value（The call returns successfully） func 0-Claw enable；1-Initialization of jaws；2-Location settings；3-Speed ​​settings；4-Torque setting；6-Read the jaw status；7-Read the initialization state；8-Read the fault code；9-Reading location；10-Reading speed；11-Read torque
    """
    @log_call
    @xmlrpc_timeout
    def GetAxleLuaGripperFunc(self,id):
        id=int(id)
        error = self.robot.GetAxleLuaGripperFunc(id)
        if error[0] == 0:
            par = error[1].split(',')
            print(len(par))
            if 16 != len(par):
                self.log_error("GetAxleLuaEnableDevice fail")
                return -1
            else:
                return (error[0], [par[0],par[1],par[2], par[3], par[4], par[5], par[6], par[7], par[8],
                        par[9], par[10], par[11], par[12], par[13], par[14], par[15]])
        else:
            return error

    """   
    @brief  Setting up controller peripheral protocolLUAfile name
    @param  [in]  id Agreement number
    @param  [in] name luaFile name “CTRL_LUA_test.lua”
    @return Error code success- 0, fail-Error code        
    """
    @log_call
    @xmlrpc_timeout
    def SetCtrlOpenLUAName(self,id,name):
        id = int(id)
        name = str(name)
        error = self.robot.SetCtrlOpenLUAName(id,name)
        return error

    """   
    @brief  Get the currently configured controller peripheral protocolLUAfile name
    @return Error code success- 0, fail-Error code     
    @return Return value（The call returns successfully） name[4] luaFile name “CTRL_LUA_test.lua”
    """
    @log_call
    @xmlrpc_timeout
    def GetCtrlOpenLUAName(self):
        error = self.robot.GetCtrlOpenLUAName()
        if error[0] == 0:
            par = error[2].split(',')
            if 4 != sizeof(par):
                self.log_error("GetCtrlOpenLUAName fail")
                return -1
            else:
                return error[0], [error[1], error[2], error[3], error[4]]
        else:
            return error

    """   
    @brief  Loading the controllerLUAprotocol
    @param  [in] id ControllerLUAAgreement number
    @return Error code success- 0, fail-Error code     
    """

    @log_call
    @xmlrpc_timeout
    def LoadCtrlOpenLUA(self,id):
        id = int(id)
        error = self.robot.LoadCtrlOpenLUA(id)
        return error

    """   
    @brief   Uninstall the controllerLUAprotocol
    @param  [in] id ControllerLUAAgreement number
    @return Error code success- 0, fail-Error code     
    """

    @log_call
    @xmlrpc_timeout
    def UnloadCtrlOpenLUA(self,id):
        id = int(id)
        error = self.robot.UnloadCtrlOpenLUA(id)
        return error

    """   
    @brief  Set up the controllerLUAProtocol Error Code
    @param  [in] id ControllerLUAAgreement number
    @return Error code success- 0, fail-Error code     
    """

    @log_call
    @xmlrpc_timeout
    def SetCtrlOpenLuaErrCode(self,id):
        id = int(id)
        error = self.robot.SetCtrlOpenLuaErrCode(id)
        return error

    """   
    @brief  robotEthercatSlave file writing
    @param  [in] type Slave file type，1-Upgrade slave files；2-Upgrade slave configuration file
    @param  [in] slaveID Slave number
    @param  [in] fileName Upload file name
    @return Error code success- 0, fail-Error code     
    """

    @log_call
    @xmlrpc_timeout
    def SlaveFileWrite(self,type,slaveID,fileName):
        type = int(type)
        slaveID = int(slaveID)
        fileName =str(fileName)
        error = self.robot.SlaveFileWrite(type,slaveID,fileName)
        return error

    """   
    @brief  robotEthercatEnter from the stationbootmodel
    @return Error code success- 0, fail-Error code     
    """

    @log_call
    @xmlrpc_timeout
    def SetSysServoBootMode(self):
        error = self.robot.SetSysServoBootMode()
        return error

    """   
    @brief  Upload endLuaOpen Agreement Document
    @param  [in] filePath localluaFile path name ".../AXLE_LUA_End_DaHuan.lua"
    @return Error code success- 0, fail-Error code     
    """

    @log_call
    @xmlrpc_timeout
    def AxleLuaUpload(self,filePath):

        error = self.__FileUpLoad(10,filePath)
        file_name = "/tmp/" + os.path.basename(filePath)
        # file_name = os.path.basename(filePath)
        if 0!= error :
            return error
        else:
            rtn = self.SetAxleFileType(2)
            if(rtn!=0):
                return -1
            rtn = self.SetSysServoBootMode()
            if(rtn!=0):
                return -1
            rtn = self.SlaveFileWrite(1,7,file_name)
            if(rtn!=0):
                return -1
            return rtn


    """   
    ***************************************************************************New********************************************************************************************
    """

    """   
    @brief  Movable device enable
    @param  [in] enable Enable status，0-Go to enable， 1-Enable
    @return Error code success- 0, fail-Error code     
    """

    @log_call
    @xmlrpc_timeout
    def TractorEnable(self, enable):
        enable = int(enable)
        error = self.robot.TractorEnable(enable)
        return error

    """   
    @brief  Redirectable device back to zero
    @return Error code success- 0, fail-Error code     
    """

    @log_call
    @xmlrpc_timeout
    def TractorHoming(self):
        error = self.robot.TractorHoming()
        return error

    """   
    @brief Moveable device linear motion
    @param [in] distance Straight line motion distance（mm）
    @param [in] vel Percentage of linear motion speed（0-100）
    @return Error code success- 0, fail-Error code  
    """

    @log_call
    @xmlrpc_timeout
    def TractorMoveL(self,distance,vel):
        if self.GetSafetyCode() != 0:
            return self.GetSafetyCode()
        distance = float(distance)
        vel = float(vel)
        error = self.robot.TractorMoveL(distance,vel)
        return error

    """   
    @brief Arc motion of movable device
    @param [in] radio Arc radius of motion（mm）
    @param [in] angle Arc motion angle（°）
    @param [in] vel Percentage of arc movement speed（0-100）
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout
    def TractorMoveC(self,radio, angle, vel):
        if self.GetSafetyCode() != 0:
            return self.GetSafetyCode()
        radio = float(radio)
        angle = float(angle)
        vel = float(vel)
        error = self.robot.TractorMoveC(radio, angle, vel)
        return error

    """   
    @brief The movable device stops movement
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout

    def TractorStop(self):
        error = self.robot.ProgramStop()
        return error

    """   
    @brief Set up wire positioning expansionIOport
    @param [in] searchDoneDINum The welding wire positioning is successfulDOport(0-127)
    @param [in] searchStartDONum Wire position search start and stop controlDOport(0-127)
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout

    def SetWireSearchExtDIONum(self,searchDoneDINum,searchStartDONum):
        searchDoneDINum = int(searchDoneDINum)
        searchStartDONum = int(searchStartDONum)
        error = self.robot.SetWireSearchExtDIONum(searchDoneDINum,searchStartDONum)
        return error

    """   
    @brief Set welding machine control mode extensionDOport
    @param [in] DONum Welding machine control modeDOport(0-127)
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout

    def SetWeldMachineCtrlModeExtDoNum(self, DONum):
        DONum = int(DONum)
        error = self.robot.SetWeldMachineCtrlModeExtDoNum(DONum)
        return error

    """   
    @brief Set the welding machine control mode
    @param [in] mode Welding machine control mode;0-Universal
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout

    def SetWeldMachineCtrlMode(self, mode):
        mode = int(mode)
        error = self.robot.SetWeldMachineCtrlMode(mode)
        return error

    """   
    @brief closureRPC
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout

    def CloseRPC(self):
        # Set a stop event to notify thread stop
        self.stop_event.set()

        # If the thread is still running，Wait for it to end
        # if self.thread.is_alive():
        #     self.thread.join()

        # Clean up XML-RPC acting
        if self.robot is not None:
            self.robot = None  # Set the proxy to None，Free up resources
            self.sock_cli_state.close()
            self.sock_cli_state = None
            self.robot_state_pkg = None
            self.closeRPC_state = True
            # self.robot_realstate_exit = False

        # If the thread is still running，Wait for it to end
        if self.thread.is_alive():
            self.thread.join()


        print("RPC connection closed.")
        return

    """   
    @brief Record teaching points
    @param [in] name Teaching point name
    @param [in] update_allprogramfile Whether to cover 0-Not covered 1-cover
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout

    def SavePoint(self,name,update_allprogramfile=0):
        name = str(name)
        update_allprogramfile = int(update_allprogramfile)
        error = self.robot.save_point(name,update_allprogramfile)
        return error

    """   
    @brief Starting strange position protection
    @param [in] protectMode Strange protection mode，0：Joint pattern；1-Cartesian mode
    @param [in] minShoulderPos Singular shoulder adjustment range(mm), default100.0
    @param [in] minElbowPos Elbow singular adjustment range(mm), default50.0
    @param [in] minWristPos Wrist-single adjustment range(°), default10.0
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout

    def SingularAvoidStart(self, protectMode, minShoulderPos=100,minElbowPos=50,minWristPos=10):
        protectMode = int(protectMode)
        minShoulderPos = float(minShoulderPos)
        minElbowPos = float(minElbowPos)
        minWristPos = float(minWristPos)
        error = self.robot.SingularAvoidStart(protectMode, minShoulderPos,minElbowPos,minWristPos)
        return error

    """   
    @brief Stop odd position protection
    @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout

    def SingularAvoidEnd(self):
        error = self.robot.SingularAvoidEnd()
        return error

    """   
        @brief Get the number of rotations of the rotating jaw
        @return Error code success- 0, fail-Error code
        @return Return value（The call returns successfully） fault 0-No errors，1-There is an error
        @return Return value（The call returns successfully） num Number of rotations
    """

    @log_call
    @xmlrpc_timeout

    def GetGripperRotNum(self):
        return 0,self.robot_state_pkg.gripper_fault,self.robot_state_pkg.gripperRotNum

    """   
        @brief Get the rotation speed percentage of the rotating jaw
        @return Error code success- 0, fail-Error code
        @return Return value（The call returns successfully） fault 0-No errors，1-There is an error
        @return Return value（The call returns successfully） speed Percent rotation speed
    """

    @log_call
    @xmlrpc_timeout

    def GetGripperRotSpeed(self):
        return 0, self.robot_state_pkg.gripper_fault, self.robot_state_pkg.gripperRotSpeed

    """   
        @brief Get the rotation torque percentage of the rotating jaw
        @return Error code success- 0, fail-Error code
        @return Return value（The call returns successfully） fault 0-No errors，1-There is an error
        @return Return value（The call returns successfully） torque Percent rotation torque
    """

    @log_call
    @xmlrpc_timeout

    def GetGripperRotTorque(self):
        return 0, self.robot_state_pkg.gripper_fault, self.robot_state_pkg.gripperRotTorque

    """   
       @brief startPtpsportsFIRFiltering
       @param  [in] maxAcc Maximum acceleration extreme value(deg/s2)
       @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout

    def PtpFIRPlanningStart(self, maxAcc):
        maxAcc = float(maxAcc)
        error = self.robot.PtpFIRPlanningStart(maxAcc)
        return error

    """   
       @brief closurePtpsportsFIRFiltering
       @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout

    def PtpFIRPlanningEnd(self):
        error = self.robot.PtpFIRPlanningEnd()
        return error

    """   
       @brief Upload trackJdocument
       @param  [in] filePath The full path name of the upload track file   C://test/testJ.txt
       @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout

    def TrajectoryJUpLoad(self,filePath):
        error = self.__FileUpLoad(20, filePath)
        return error

    """2024.12.16"""
    """   
       @brief Delete TracksJdocument
       @param  [in] filePath Delete the full path name of the track file   C://test/testJ.txt
       @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout

    def TrajectoryJDelete(self, fileName):
        error = self.__FileDelete(20, fileName)
        return error

    """2024.12.18"""
    """   
       @brief startLIN、ARCsportsFIRFiltering
       @param  [in] maxAccLin Linear acceleration extreme value(mm/s2)
       @param  [in] maxAccDeg Angle acceleration extreme value(deg/s2)
       @param  [in] maxJerkLin Linear acceleration extreme value(mm/s3)
       @param  [in] maxJerkDeg Angle acceleration extreme value(deg/s3)
       @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout

    def LinArcFIRPlanningStart(self, maxAccLin, maxAccDeg, maxJerkLin, maxJerkDeg):
        maxAccLin = float(maxAccLin)
        maxAccDeg = float(maxAccDeg)
        maxJerkLin = float(maxJerkLin)
        maxJerkDeg = float(maxJerkDeg)
        error = self.robot.LinArcFIRPlanningStart(maxAccLin, maxAccDeg, maxJerkLin, maxJerkDeg)
        return  error

    """   
       @brief closureLIN、ARCsportsFIRFiltering
       @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout

    def LinArcFIRPlanningEnd(self):
        error = self.robot.LinArcFIRPlanningEnd()
        return error

    """2025.01.08"""
    """   
       @brief Tool coordinate system conversion starts
       @param  [in] toolNum Tool coordinate system number[0-14]
       @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout

    def ToolTrsfStart(self, toolNum):
        toolNum = int(toolNum)
        error = self.robot.ToolTrsfStart(toolNum)
        return error

    """   
       @brief Tool coordinate system conversion ends
       @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout

    def ToolTrsfEnd(self):
        error = self.robot.ToolTrsfEnd()
        return error

    """2025.01.08"""
    """3.7.8"""
    """
       @brief Calculate the tool coordinate system based on point information
       @param  [in] method Calculation method；0-Four-point method；1-Six-point method
       @param  [in] pos Joint position group，The length of the array is4indivual，The length of the array is6indivual
       @return Error code success- 0, fail-Error code
       @return Return value（The call returns successfully） tcp_offset=[x,y,z,rx,ry,rz]: Tool coordinate system calculated based on point information，unit [mm][°]
    """

    @log_call
    @xmlrpc_timeout

    def ComputeToolCoordWithPoints(self, method, pos):
        method = int(method)
        param = {}
        param[0] = pos[0]
        param[1] = pos[1]
        param[2] = pos[2]
        param[3] = pos[3]

        if method == 0:  # Four-point method
            param[4] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            param[5] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        else:  # Six-point method
            param[4] = pos[4]
            param[5] = pos[5]
        _error = self.robot.ComputeToolCoordWithPoints(method, param[0], param[1], param[2], param[3], param[4], param[5])
        error = _error[0]
        if error == 0:
            return error, [_error[1], _error[2], _error[3], _error[4], _error[5], _error[6]]
        return error

    """
       @brief Calculate the workpiece coordinate system based on point information
       @param  [in] method Calculation method；0：origin-xaxis-zaxis  1：origin-xaxis-xyflat
       @param  [in] pos ThreeTCPLocation group
       @param  [in] refFrame Reference coordinate system
       @return Error code success- 0, fail-Error code
       @return Return value（The call returns successfully） wobj_offset=[x,y,z,rx,ry,rz]: Workpiece coordinate system calculated based on point information，unit [mm][°]
    """

    @log_call
    @xmlrpc_timeout

    def ComputeWObjCoordWithPoints(self, method, pos, refFrame):
        method = int(method)
        param = {}
        param[0] = pos[0]
        param[1] = pos[1]
        param[2] = pos[2]
        refFrame = int(refFrame)
        _error = self.robot.ComputeWObjCoordWithPoints(method, param[0], param[1], param[2], refFrame)
        error = _error[0]
        if error == 0:
            return error, [_error[1], _error[2], _error[3], _error[4], _error[5], _error[6]]
        return error

    """
       @brief Set the detection parameters for robot welding arc accident interrupt
       @param  [in] checkEnable Whether to enable detection；0-Not enabled；1-Enable
       @param  [in] arcInterruptTimeLength Arc interrupt confirmation time(ms)
       @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout

    def WeldingSetCheckArcInterruptionParam(self, checkEnable, arcInterruptTimeLength):
        checkEnable = int(checkEnable)
        arcInterruptTimeLength = int(arcInterruptTimeLength)
        error = self.robot.WeldingSetCheckArcInterruptionParam(checkEnable, arcInterruptTimeLength)
        return error

    """
       @brief Obtain the detection parameters of robot welding arc accidental interrupt
       @return Error code success- 0, fail-Error code
       @return Return value（The call returns successfully） checkEnable Whether to enable detection；0-Not enabled；1-Enable
       @return Return value（The call returns successfully） arcInterruptTimeLength Arc interrupt confirmation time(ms)
    """

    @log_call
    @xmlrpc_timeout

    def WeldingGetCheckArcInterruptionParam(self):
        _error = self.robot.WeldingGetCheckArcInterruptionParam()
        error = _error[0]
        if error == 0:
            return error, _error[1], _error[2]
        return error

    """
       @brief Set the robot welding interrupt recovery parameters
       @param  [in] enable Whether to enable welding interrupt recovery
       @param  [in] length Weld overlap distance(mm)
       @param  [in] velocity The percentage of speed of the robot returning to the arc point again(0-100)
       @param  [in] moveType How the robot moves to the arc point again；0-LIN；1-PTP
       @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout

    def WeldingSetReWeldAfterBreakOffParam(self, enable, length, velocity, moveType):
        enable = int(enable)
        length = float(length)
        velocity = float(velocity)
        moveType = int(moveType)
        error = self.robot.WeldingSetReWeldAfterBreakOffParam(enable, length, velocity, moveType)
        return error

    """
       @brief Obtain the robot welding interrupt recovery parameters
       @return Error code success- 0, fail-Error code
       @return Return value（The call returns successfully） enable Whether to enable welding interrupt recovery
       @return Return value（The call returns successfully） length Weld overlap distance(mm)
       @return Return value（The call returns successfully） velocity The percentage of speed of the robot returning to the arc point again(0-100)
       @return Return value（The call returns successfully） moveType How the robot moves to the arc point again；0-LIN；1-PTP
    """

    @log_call
    @xmlrpc_timeout

    def WeldingGetReWeldAfterBreakOffParam(self):
        _error = self.robot.WeldingGetReWeldAfterBreakOffParam()
        error = _error[0]
        if error == 0:
            return error, _error[1], _error[2], _error[3], _error[4]
        return error

    """
       @brief Set up the robot welding to resume welding after interruption
       @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout

    def WeldingStartReWeldAfterBreakOff(self):
        error = self.robot.WeldingStartReWeldAfterBreakOff()
        return error

    """
       @brief Set the robot to exit welding after interruption
       @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout

    def WeldingAbortWeldAfterBreakOff(self):
        error = self.robot.WeldingAbortWeldAfterBreakOff()
        return error

    """2025.01.09"""
    """
       @brief 
       @param  [in] status
       @param  [in] delayMode
       @param  [in] delayTime
       @param  [in] delayDisExAxisNum
       @param  [in] delayDis
       @param  [in] sensitivePara
       @param  [in] speed
       @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout

    def LaserSensorRecord(self, status, delayMode, delayTime, delayDisExAxisNum, delayDis, sensitivePara, speed):
        status = int(status)
        delayMode = int(delayMode)
        delayTime = int(delayTime)
        delayDisExAxisNum = int(delayDisExAxisNum)
        delayDis = float(delayDis)
        sensitivePara = float(sensitivePara)
        speed = float(speed)
        error = self.robot.LaserSensorRecord(status, delayMode, delayTime, delayDisExAxisNum, delayDis, sensitivePara, speed)
        return error

    """
       @brief 
       @param  [in] weldId
       @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout

    def LaserTrackingLaserOn(self, weldId):
        weldId = int(weldId)
        error = self.robot.LaserTrackingLaserOn(weldId)
        return error

    """
       @brief 
       @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout

    def LaserTrackingLaserOff(self):
        error = self.robot.LaserTrackingLaserOff()
        return error

    """
       @brief 
       @param  [in] coordId
       @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout

    def LaserTrackingTrackOn(self, coordId):
        coordId = int(coordId)
        error = self.robot.LaserTrackingTrackOn(coordId)
        return error

    """
       @brief 
       @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout

    def LaserTrackingTrackOff(self):
        error = self.robot.LaserTrackingTrackOff()
        return error

    """
       @brief 
       @param  [in] direction
       @param  [in] directionPoint
       @param  [in] vel
       @param  [in] distance
       @param  [in] timeout
       @param  [in] posSensorNum
       @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout

    def LaserTrackingSearchStart(self, direction, directionPoint, vel, distance, timeout, posSensorNum):
        direction = int(direction)
        directionPoint = list(map(float, directionPoint))
        vel = int(vel)
        distance = int(distance)
        timeout = int(timeout)
        posSensorNum = int(posSensorNum)
        error = self.robot.LaserTrackingSearchStart(direction, directionPoint, vel, distance, timeout, posSensorNum)
        return error

    """
       @brief 
       @return Error code success- 0, fail-Error code
    """

    @log_call
    @xmlrpc_timeout

    def LaserTrackingSearchStop(self):
        error = self.robot.LaserTrackingSearchStop()
        return error
