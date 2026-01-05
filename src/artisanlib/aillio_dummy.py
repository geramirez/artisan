#!/usr/bin/env python3

# ABOUT
# Aillio "Dummy" support for Artisan

# This device is designed to build simulations for the r2 to allow people 
# experiment with roasting

# LICENSE
# This program or module is free software: you can redistribute it and/or
# modify it under the terms of the GNU General Public License as published
# by the Free Software Foundation, either version 2 of the License, or
# version 3 of the License, or (at your option) any later version. It is
# provided for educational purposes and is distributed in the hope that
# it will be useful, but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See
# the GNU General Public License for more details.

# AUTHOR
# mikefsq, r2 2025

import time
import json
import threading
from struct import unpack
from multiprocessing import Pipe
from platform import system
from collections.abc import Generator

import usb.core # type: ignore[import-untyped]
import usb.util # type: ignore[import-untyped]

import numpy as np

if system().startswith('Windows'):
    import libusb_package # pyright:ignore[reportMissingImports] # pylint: disable=import-error # ty:ignore[unresolved-import]


import logging
import random
from typing import Final, Any, TypedDict, TYPE_CHECKING

if TYPE_CHECKING:
    try:
        from multiprocessing.connection import PipeConnection as Connection # type: ignore[unused-ignore,attr-defined,assignment] # pylint: disable=unused-import
    except ImportError:
        from multiprocessing.connection import Connection # type: ignore[unused-ignore,attr-defined,assignment] # pylint: disable=unused-import
    from usb.core import Configuration, Interface, Endpoint

_log: Final[logging.Logger] = logging.getLogger(__name__)


def _load_library(find_library:Any = None) -> Any:
    import usb.libloader # type: ignore[import-untyped, unused-ignore] # pylint: disable=redefined-outer-name
    return usb.libloader.load_locate_library(
                ('usb-1.0', 'libusb-1.0', 'usb'),
                'cygusb-1.0.dll', 'Libusb 1',
                find_library=find_library, check_symbols=('libusb_init',))

class DEVICE_VARIANT(TypedDict):
    vid: int
    pid: int
    protocol: int
    model: str

class AillioDummy:
    AILLIO_INTERFACE = 0x1
    AILLIO_CONFIGURATION = 0x1
    AILLIO_DEBUG = 1
    AILLIO_CMD_INFO1 = [0x30, 0x02]
    AILLIO_CMD_INFO2 = [0x89, 0x01]
    AILLIO_CMD_STATUS1 = [0x30, 0x01]
    AILLIO_CMD_STATUS2 = [0x30, 0x03]
    AILLIO_CMD_PRS = [0x30, 0x01, 0x00, 0x00]
    AILLIO_CMD_HEATER_INCR = [0x34, 0x01, 0xaa, 0xaa]
    AILLIO_CMD_HEATER_DECR = [0x34, 0x02, 0xaa, 0xaa]
    AILLIO_CMD_DRUM_INCR = [0x32, 0x01, 0xaa, 0xaa]
    AILLIO_CMD_DRUM_DECR = [0x32, 0x02, 0xaa, 0xaa]
    AILLIO_CMD_FAN_INCR = [0x31, 0x01, 0xaa, 0xaa]
    AILLIO_CMD_FAN_DECR = [0x31, 0x02, 0xaa, 0xaa]

    AILLIO_STATE_OFF = 0x00
    AILLIO_STATE_PH = 0x02
    AILLIO_STATE_STABILIZING = 0x03
    AILLIO_STATE_READY_TO_ROAST = 0x04
    AILLIO_STATE_CHARGE = 0x04
    AILLIO_STATE_ROASTING = 0x06
    AILLIO_STATE_COOLDOWN = 0x07
    AILLIO_STATE_COOLING = 0x08
    AILLIO_STATE_SHUTDOWN = 0x09
    AILLIO_STATE_BACK_TO_BACK = 0x08
    AILLIO_STATE_POWER_ON_RESET = 0x0B

	# sequence of valid states in PRS Button sequence
    VALID_STATES = [
            AILLIO_STATE_OFF,
            AILLIO_STATE_PH,
            AILLIO_STATE_CHARGE,
            AILLIO_STATE_ROASTING,
            AILLIO_STATE_COOLDOWN
    ]

    FRAME_TYPES = {
        0xA0: 'Temperature Frame',
        0xA1: 'Fan Control Frame',
        0xA2: 'Power Frame',
        0xA3: 'A3 Frame',
        0xA4: 'A4 Frame',
    }

    FRAME_SIZE = 64

    def __init__(self, debug:bool = False) -> None:
        # rewards
        self.rewards = 0

        # Thread safety and cleanup controls
        self._cleanup_lock = threading.Lock()
        self._is_cleanup_done:bool = False
        self.worker_thread_run:bool = True

        # Communication pipes
        self.parent_pipe: Connection|None = None # type:ignore[no-any-unimported, unused-ignore]
        self.child_pipe: Connection|None = None # type:ignore[no-any-unimported, unused-ignore]
        self.worker_thread: threading.Thread|None = None

        # Basic configuration
        self.simulated:bool = False
        self.AILLIO_DEBUG = True
        self.__dbg('init')

        # USB handling
        self.usbhandle:Generator[usb.core.Device, Any, None]|usb.core.Device|None = None # type:ignore[no-any-unimported,unused-ignore]
        self.protocol:int = 2
        self.model:str = 'Unknown'
        self.ep_in:Endpoint|None = None  # type:ignore[no-any-unimported]
        self.ep_out:Endpoint|None = None # type:ignore[no-any-unimported]
        self.TIMEOUT = 1000  # USB timeout in milliseconds
        self.FRAME_SIZE = 64  # Standard USB packet size

        # Device variants
        self.DEVICE_VARIANTS:list[DEVICE_VARIANT] = [
            {'vid': 0x0483, 'pid': 0xa4cd, 'protocol': 2, 'model': 'Aillio Bullet Dummy'},
        ]

        # Fields from R1 for compatibility
        self.bt:float = 0
        self.dt:float = 0
#        self.heater:float = 0
#        self.fan:float = 0
        self.bt_ror:float = 0
        self.dt_ror:float = 0
#        self.drum:float = 0
        self.voltage:float = 0
#        self.exitt:float = 0
#        self.state_str:str = ''
#        self.r1state:int = 0
        self.roast_number:int = -1
        self.fan_rpm:float = 0
        self.irt:float = 0
        self.pcbt:float = 0
#        self.coil_fan:int = 0
#        self.coil_fan2:int = 0
        self.pht:int = 0
#        self.minutes = 0
#        self.seconds = 0

        # A0 Frame - Basic Temperature Data
        self.ibts_bean_temp: float = 30.0
        self.ibts_bean_temp_rate: float = 0.0
        self.ibts_ambient_temp: float = 0.0
        self.bean_probe_temp: float = 30.0
        self.bean_probe_temp_rate: float = 0.0
        self.energy_used_this_roast: float = 0.0
        self.differential_air_pressure: float = 0.0
        self.exhaust_fan_rpm: int = 0
        self.inlet_air_temp: float = 0.0
        self.hot_air_temp: float = 0.0
        self.exitt: float = 0.0  # ExhaustAirTemp
        self.absolute_atmospheric_pressure: float = 0.0
        self.humidity_roaster: float = 0.0
        self.humidity_temp: float = 0.0
        self.minutes: int = 0
        self.button_crack_mark: int = 0
        self.seconds: int = 0
        self.heater: int = 0    # PSet - start at high for realistic roast timing
        self.fan: int = 0        # FSet - start at medium-high for good heat transfer
        self.ha_set: int = 0     # HASet
        self.drum: int = 7       # DSet - start at medium-high
        self.r_count: int = 0
        self.fc_sample_index: int = 0
        self.fc_number_cracks: int = 0
        self.roasting_method: int = 0  # Manual or Recipe
        self.status: int = 0
        self.r1state: int = 0   # StateMachine

        # A1 Frame - Fan Control and Error Data
        self.error_counts: int = 0
        self.extra_u8_1: int = 0
        self.extra_u8_2: int = 0
        self.extra_u8_3: int = 0
        self.error_category1: int = 0
        self.error_type1: int = 0
        self.error_info1: int = 0
        self.error_value1: int = 0
        self.error_category2: int = 0
        self.error_type2: int = 0
        self.error_info2: int = 0
        self.error_value2: int = 0
        self.coil_fan1_duty: int = 0
        self.coil_fan2_duty: int = 0
        self.induction_fan1_duty: int = 0
        self.induction_fan2_duty: int = 0
        self.induction_blower_duty: int = 0
        self.icf1_fan_duty: int = 0
        self.exhaust_blower_duty: int = 0
        self.vibrator_motor_duty: int = 0
        self.reserved_u16_1: int = 0
        self.coil_fan: int = 0  # CoilFan1Rpm
        self.coil_fan2: int = 0  # CoilFan2Rpm
        self.induction_fan1_rpm: int = 0
        self.induction_fan2_rpm: int = 0
        self.induction_blower_rpm: int = 0
        self.icf1_fan_rpm: int = 0
        self.ibts_fan_rpm: int = 0
        self.roast_drum_rpm: int = 0
        self.reserved_u16_3: int = 0
        self.control_board_critical: int = 0
        self.buttons: int = 0

        # A2 Frame - Power Data
        self.power_setpoint_watt: int = 0
        self.line_frequency_hz: float = 0.0
        self.igbt_frequency_hz: int = 0
        self.igbt_error: int = 0
        self.igbt_temp1: float = 0.0
        self.igbt_temp2: float = 0.0
        self.state_runtime: int = 0
        self.status_register: int = 0
        self.status_error: int = 0
        self.voltage_rms: float = 0.0
        self.current_rms: float = 0.0
        self.active_power: float = 0.0
        self.reactive_power: float = 0.0
        self.apparent_power: float = 0.0
        self.accumulator_energy: int = 0
        self.extra_u16_1: int = 0
        self.extra_u16_2: int = 0
        self.extra_u16_3: int = 0
        self.extra_u16_4: int = 0
        self.extra_u8_5: int = 0
        self.extra_u8_6: int = 0
        self.extra_u32_7: int = 0
        self.extra_f32_8: float = 0.0
        self.extra_f32_9: float = 0.0
        self.extra_f32_10: float = 0.0

        # Other fields
        self.state_str: str = ''

        # Roasting simulation state
        self.roast_time: float = 0.0  # Total roast time in seconds
        self.first_crack_occurred: bool = False
        self.second_crack_occurred: bool = False
        self.development_time_start: float = 0.0  # Time when first crack starts
        self.moisture_content: float = 12.0  # Initial moisture percentage
        self.roast_phase: str = 'drying'  # drying, maillard, development, cooling

        # Initialize USB connection and start worker thread
        self._open_port()

    def __del__(self) -> None:
        if not self.simulated:
            self._close_port()

    def __dbg(self, msg:str) -> None:
        _log.debug('Aillio: %s', msg)
        if self.AILLIO_DEBUG and not self.simulated:
            try:
                print('AillioDummy: ' + msg)
            except OSError:
                pass

    def _open_port(self) -> None:
        self.__dbg('connected to dummy device')

    def _close_port(self) -> None:
        self.__dbg('disconnect to dummy device')
    
    def __sendcmd(self, cmd:list[int]|bytes) -> None:
        self.__dbg('sending command: ' + str(cmd))
        if self.usbhandle is None or self.ep_out is None:
            raise OSError('Device not properly initialized')

        try:
            if isinstance(cmd, list):
                cmd = bytes(cmd)

            self.ep_out.write(cmd, timeout=self.TIMEOUT)

        except Exception as e: # pylint: disable=broad-except
            raise OSError(f'Failed to send command: {str(e)}') from e

    def __readreply(self, length:int) -> Any:
        if self.usbhandle is None or self.ep_in is None:
            raise OSError('Device not properly initialized')
        try:
            packets_needed = (length + self.FRAME_SIZE - 1) // self.FRAME_SIZE
            total_length = packets_needed * self.FRAME_SIZE

            data = self.ep_in.read(total_length, timeout=self.TIMEOUT)
            return data[:length]
        except Exception as e: # pylint: disable=broad-except
            raise OSError(f'Failed to read reply: {str(e)}') from e

    @staticmethod
    def __debug_frame(data: bytes) -> None:
        for i in range(0, len(data), 16):
            chunk = data[i:i+16]
            hex_values = ' '.join(f'{b:02x}' for b in chunk)
            ascii_values = ''.join(chr(b) if 32 <= b <= 126 else '.' for b in chunk)
            print(f"{i:04x}: {hex_values:<48} {ascii_values}")


    def tick(self, action: int = 0) -> None:
        """Simulate one time step (1 second) of roasting with improved physics."""
        dt = 1.0  # Time step in seconds
        
        # Apply action to heater (heater range is 0-14)
        self.set_heater(np.clip(self.get_heater() + action, 0, 14))
        
        # Store previous values
        previous_ror = self.ibts_bean_temp_rate
        previous_temp = self.ibts_bean_temp
        previous_bean_probe = self.bean_probe_temp
        
        # Calculate air temperature from heater and fan settings
        air_temp = self.calculate_air_temp()
        
        # Heat transfer coefficient affected by fan speed
        heat_transfer_coeff = self.calculate_heat_transfer(self.fan)
        
        # Endothermic/exothermic effects from chemical reactions
        reaction_heat = self.calculate_reaction_heat(self.ibts_bean_temp, self.roast_time)
        
        # Temperature change with realistic physics
        dT_ibts = heat_transfer_coeff * (air_temp - self.ibts_bean_temp) + reaction_heat
        dT_ibts += np.random.normal(0, 0.05)  # Bidirectional sensor noise
        
        # Bean probe has more thermal lag
        dT_probe = heat_transfer_coeff * 0.8 * (air_temp - self.bean_probe_temp) + reaction_heat * 0.9
        dT_probe += np.random.normal(0, 0.05)
        
        # Update temperatures
        self.ibts_bean_temp += dT_ibts * dt
        self.bean_probe_temp += dT_probe * dt
        
        # Clamp to realistic ranges
        self.ibts_bean_temp = np.clip(self.ibts_bean_temp, 0, 260)
        self.bean_probe_temp = np.clip(self.bean_probe_temp, 0, 260)
        
        # Calculate RoR in °C/min
        self.ibts_bean_temp_rate = (self.ibts_bean_temp - previous_temp) * 60
        self.bean_probe_temp_rate = (self.bean_probe_temp - previous_bean_probe) * 60
        
        # Update roast time and moisture content
        self.roast_time += dt
        self.update_moisture_content(dt)
        
        # Update roast phase and detect crack events
        self.update_roast_phase()
        
        # Calculate rewards based on roast profile quality
        self.rewards += self.calculate_rewards(previous_ror)

    def calculate_rewards(self, prev_ror: float) -> float:
        """Calculate stage-aware rewards for RL training."""
        reward = 0.0
        ror = self.ibts_bean_temp_rate
        ror_change = ror - prev_ror
        temp = self.ibts_bean_temp
        
        # Penalize dangerous conditions
        if temp > 240:
            reward -= 20  # Burning territory
        elif temp > 230:
            reward -= 5  # Getting too hot
        
        # Stage-specific RoR targets
        if self.roast_phase == 'drying':
            # Want steady RoR increase in drying phase (100-160°C)
            target_ror = 8.0
            if 6 <= ror <= 12:
                reward += 5
            if abs(ror_change) < 1.0:  # Steady is good
                reward += 2
        
        elif self.roast_phase == 'maillard':
            # Want decreasing RoR but controlled (160-190°C)
            target_ror = 6.0
            if 4 <= ror <= 8:
                reward += 5
            if -2 <= ror_change <= 0:  # Gentle decrease
                reward += 3
            elif ror_change < -5:  # Too fast decrease
                reward -= 5
        
        elif self.roast_phase == 'development':
            # Post first crack - want steady moderate RoR (190-220°C)
            target_ror = 5.0
            if 3 <= ror <= 7:
                reward += 8
            if abs(ror_change) < 0.5:  # Very steady is critical
                reward += 5
            if ror_change > 2:  # Acceleration after FC is bad
                reward -= 10
        
        # Penalize stalling/negative RoR before development
        if self.roast_phase != 'development' and ror < 0:
            reward -= 10
        
        # Penalize extreme RoR changes (crash/flick)
        if abs(ror_change) > 10:
            reward -= 15
        
        # Bonus for reaching first crack at reasonable temp
        if self.first_crack_occurred and 190 <= temp <= 205:
            reward += 10
        
        return reward

    def calculate_air_temp(self) -> float:
        """Calculate chamber air temperature from heater and fan settings."""
        # Base air temperature from heater power
        # Roasting chamber air is MUCH hotter than beans to drive heat transfer
        # Realistic range: heater 0-14 -> 200-470°C chamber air temp
        base_temp = 200 + self.heater * 20
        
        # Fan cooling effect (more fan = more air flow = cooler effective temp)
        # Higher fan speeds remove more heat from the bean environment
        cooling_effect = self.fan * 4
        
        return base_temp - cooling_effect

    def calculate_heat_transfer(self, fan_speed: float) -> float:
        """Calculate heat transfer coefficient based on fan speed.
        
        Higher fan speed = better convection = faster heat transfer.
        Calibrated for realistic roasting: ~8-12 minutes to reach 195°C (first crack)
        at high heater settings, ~6+ minutes to reach 260°C at maximum.
        """
        # Base heat transfer coefficient tuned for realistic timescale
        # Adjusted to prevent stalling at 130°C while keeping total time around 6-7 min
        base_coeff = 0.0006
        
        # Fan contribution to heat transfer
        fan_contribution = fan_speed * 0.0002
        
        return base_coeff + fan_contribution

    def calculate_reaction_heat(self, temp: float, time: float) -> float:
        """Calculate heat absorbed/released by chemical reactions during roasting.
        
        Returns temperature change rate (°C/s) from reactions.
        """
        reaction = 0.0
        
        # Water evaporation (endothermic, 100-150°C)
        # Slows heating significantly when moisture is escaping
        if 100 < temp < 150 and self.moisture_content > 3:
            evaporation_intensity = (temp - 100) / 50  # Ramps up with temp
            reaction -= 0.3 * evaporation_intensity * (self.moisture_content / 12)
        
        # First crack (endothermic, 190-205°C)
        # Beans expand and absorb heat
        if 190 < temp < 205 and not self.first_crack_occurred:
            crack_intensity = (temp - 190) / 15
            reaction -= 1.5 * crack_intensity
        elif 195 < temp < 200 and self.first_crack_occurred:
            # Tail end of first crack
            reaction -= 0.3
        
        # Maillard reactions (mildly exothermic, 140-170°C)
        if 140 < temp < 170:
            maillard_intensity = np.sin((temp - 140) * np.pi / 30)  # Peak around 155°C
            reaction += 0.15 * maillard_intensity
        
        # Caramelization (exothermic, 160-220°C)
        if 160 < temp < 220:
            caramel_intensity = (temp - 160) / 60
            reaction += 0.25 * caramel_intensity
        
        # Second crack and pyrolysis (highly exothermic, 220-240°C)
        if temp > 220:
            pyrolysis_intensity = (temp - 220) / 20
            reaction += 0.4 * pyrolysis_intensity
        
        return reaction

    def update_moisture_content(self, dt: float) -> None:
        """Update bean moisture content during roasting."""
        temp = self.ibts_bean_temp
        
        # Moisture loss rate depends on temperature
        if temp > 100:
            # More moisture lost at higher temps
            loss_rate = 0.002 * (temp - 100) / 100  # %/second
            self.moisture_content = max(0, self.moisture_content - loss_rate * dt)

    def update_roast_phase(self) -> None:
        """Update roast phase and detect crack events."""
        temp = self.ibts_bean_temp
        
        # Phase transitions based on temperature
        if temp < 160:
            self.roast_phase = 'drying'
        elif temp < 190:
            self.roast_phase = 'maillard'
        else:
            self.roast_phase = 'development'
        
        # Detect first crack
        if not self.first_crack_occurred and temp >= 195:
            self.first_crack_occurred = True
            self.development_time_start = self.roast_time
            self.__dbg(f'First crack at {temp:.1f}°C, time {self.roast_time:.0f}s')
        
        # Detect second crack
        if not self.second_crack_occurred and temp >= 220:
            self.second_crack_occurred = True
            self.__dbg(f'Second crack at {temp:.1f}°C, time {self.roast_time:.0f}s') 

    @staticmethod
    def calculate_crc32(data:bytes) -> int:
        SHORT_LOOKUP_TABLE = [
            0x00000000, 0x04c11db7, 0x09823b6e, 0x0d4326d9,
            0x130476dc, 0x17c56b6b, 0x1a864db2, 0x1e475005,
            0x2608edb8, 0x22c9f00f, 0x2f8ad6d6, 0x2b4bcb61,
            0x350c9b64, 0x31cd86d3, 0x3c8ea00a, 0x384fbdbd,
        ]

        def crc32_fast(arg1:int, arg2:int) -> int:
            rax = (arg1 ^ arg2) & 0xffffffff
            idx = (rax >> 0x1c) & 0xf
            rcx_2 = ((rax << 4) & 0xffffffff) ^ SHORT_LOOKUP_TABLE[idx]

            idx = (rcx_2 >> 0x1c) & 0xf
            rax_4 = ((rcx_2 << 4) & 0xffffffff) ^ SHORT_LOOKUP_TABLE[idx]

            idx = (rax_4 >> 0x1c) & 0xf
            rcx_6 = ((rax_4 << 4) & 0xffffffff) ^ SHORT_LOOKUP_TABLE[idx]

            idx = (rcx_6 >> 0x1c) & 0xf
            return ((rcx_6 << 4) & 0xffffffff) ^ SHORT_LOOKUP_TABLE[idx]

        data_copy = bytearray(data)
        data_copy[-4:] = [0, 0, 0, 0]

        ints:list[int] = []
        for i in range(0, len(data_copy), 4):
            val = int.from_bytes(data_copy[i:i+4], 'big')
            ints.append(val)

        result = 0xffffffff
        for val in ints:
            result = crc32_fast(result, val) & 0xffffffff

        return result & 0xffffffff

    def prepare_command(self, cmd:list[int]|bytes) -> bytes:
        if isinstance(cmd, list):
            cmd = bytes(cmd)
        cmd_with_crc = bytearray(cmd)
        cmd_with_crc.extend([0, 0, 0, 0])
        crc = self.calculate_crc32(bytes(cmd_with_crc))
        cmd_with_crc[-4:] = crc.to_bytes(4, 'little')
        return bytes(cmd_with_crc)

    def get_roast_number(self) -> int:
        return self.roast_number

    def get_bt(self) -> float:
        """Get ibts temperature in °C"""
        return self.ibts_bean_temp

    def get_dt(self) -> float:
        """Get bean probe temperature in °C"""
        return self.bean_probe_temp

    def get_heater(self) -> float:
        self.__dbg('get_heater')
        return self.heater

    def get_fan(self) -> float:
        self.__dbg('get_fan')
        return self.fan

    def get_fan_rpm(self) -> float:
        self.__dbg('get_fan_rpm')
        return self.fan_rpm

    def get_drum(self) -> float:
        return self.drum

    def get_voltage(self) -> float:
        return self.voltage_rms

    def get_bt_ror(self) -> float:
        """Get ibts temperature rate of rise in °C/min"""
        return self.ibts_bean_temp_rate

    def get_dt_ror(self) -> float:
        """Get bean probe temperature rate of rise in °C/min"""
        return self.bean_probe_temp_rate

    def get_exit_temperature(self) -> float:
        return self.exitt

    def get_state_string(self) -> str:
        return self.state_str

    def get_state(self) -> int:
        return self.r1state

    # R2 specific
    def get_humidity(self) -> float:
        return self.humidity_roaster

    def get_atmospheric_pressure(self) -> float:
        return self.absolute_atmospheric_pressure

    def get_energy_used(self) -> float:
        return self.energy_used_this_roast

    def get_crack_count(self) -> int:
        return self.fc_number_cracks

    def get_ibts_ambient_temp(self) -> float:
        return self.ibts_ambient_temp

    def set_heater(self, value:float) -> None:
        self.__dbg('set_heater ' + str(value))
        value = int(value)
        if value < 0:
            value = 0
        elif value > 14:
            value = 14

        h = self.get_heater()
        d = abs(h - value)
        if d <= 0:
            return
        d = int(float(min(d,9)))
        if h > value:
            cmd = self.prepare_command(self.AILLIO_CMD_HEATER_DECR)
            if self.parent_pipe is not None:
                for _ in range(d):
                    self.parent_pipe.send(cmd)
        else:
            cmd = self.prepare_command(self.AILLIO_CMD_HEATER_INCR)
            if self.parent_pipe is not None:
                for _ in range(d):
                    self.parent_pipe.send(cmd)
        self.heater = value

    def set_fan(self, value:float) -> None:
        self.__dbg('set_fan ' + str(value))
        value = int(value)
        if value < 0:
            value = 0
        elif value > 12:
            value = 12
        f = self.get_fan()
        d = abs(f - value)
        if d <= 0:
            return
        d = int(round(min(d,11)))
        if f > value:
            cmd = self.prepare_command(self.AILLIO_CMD_FAN_DECR)
            if self.parent_pipe is not None:
                for _ in range(d):
                    self.parent_pipe.send(cmd)
        else:
            cmd = self.prepare_command(self.AILLIO_CMD_FAN_INCR)
            if self.parent_pipe is not None:
                for _ in range(d):
                    self.parent_pipe.send(cmd)
        self.fan = value

    def set_drum(self, value:float) -> None:
        self.__dbg('set_drum ' + str(value))
        value = int(value)
        if value < 1:
            value = 1
        elif value > 9:
            value = 9

        d = self.get_drum()
        delta = abs(d - value)
        if delta <= 0:
            return
        delta = int(float(min(delta,9)))

        if d > value:
            cmd = self.prepare_command(self.AILLIO_CMD_DRUM_DECR)
            if self.parent_pipe is not None:
                for _ in range(delta):
                    self.parent_pipe.send(cmd)
        else:
            cmd = self.prepare_command(self.AILLIO_CMD_DRUM_INCR)
            if self.parent_pipe is not None:
                for _ in range(delta):
                    self.parent_pipe.send(cmd)
        self.drum = value

    def set_state(self, target_state: int) -> None:
        self.__dbg(f'set_state {target_state}')
        current_state = self.get_state()
        if current_state == target_state:
            return

        try:
            current_pos = self.VALID_STATES.index(current_state)
            target_pos = self.VALID_STATES.index(target_state)
        except ValueError:
            return

        if target_pos > current_pos:
            presses = target_pos - current_pos
        else:
            presses = len(self.VALID_STATES) - current_pos + target_pos

        for _ in range(presses):
            self.prs()
            time.sleep(0.5)

    def prs(self) -> None:
        """Press PRS button"""
        self.__dbg('PRS')
        if self.parent_pipe is not None:
            cmd = self.prepare_command(self.AILLIO_CMD_PRS)
            self.parent_pipe.send(cmd)

    def send_command(self, str_in:str) -> None:
        if str_in.startswith('send(') and str_in.endswith(')'):
            str_in = str_in[len('send('):-1]
        json_data = json.loads(str_in)
        command = json_data.get('command', '').strip().lower()

        if command == 'prs':
            self.prs()

        elif command == 'reset':
            pass

        elif command == 'start':
            current_state = self.get_state()
            if current_state in [self.AILLIO_STATE_PH, self.AILLIO_STATE_READY_TO_ROAST]:
                self.set_drum(5)
                self.set_fan(2)
                self.set_heater(2)
                self.set_state(self.AILLIO_STATE_ROASTING)
                time.sleep(0.25)
                self.set_drum(5)
                self.set_fan(2)
                self.set_heater(2)

            else:
                print('Machine must be in PH or Ready state to start preheat')

        elif command == 'dopreheat':
            if self.get_state() == self.AILLIO_STATE_OFF:
                self.set_state(self.AILLIO_STATE_PH)
            else:
                print('Machine must be in OFF state to start preheat')

        elif command in {'on', 'off', 'sync', 'charge', 'dryend', 'fcstart', 'fcend', 'scstart', 'scend'}:
            pass #usb not connected reliably at this point

        elif command == 'drop':
            current_state = self.get_state()
            if current_state == self.AILLIO_STATE_ROASTING:
                print('Starting cooling cycle...')
                self.set_state(self.AILLIO_STATE_COOLDOWN)
                print(f"New state: {self.get_state_string()}")
            else:
                print(f"Cannot start cooling from {self.get_state_string()} state")
                print('Machine must be in ROASTING state')

        elif command == 'coolend':
            current_state = self.get_state()
            print(f"Current state: {self.get_state_string()} (0x{current_state:02x})")
            if current_state in [self.AILLIO_STATE_COOLING, self.AILLIO_STATE_COOLDOWN]:
                print('Stopping roast...')
                self.set_state(self.AILLIO_STATE_OFF)
                time.sleep(0.5)
                print(f"New state: {self.get_state_string()} (0x{self.get_state():02x})")
            else:
                print(f"Cannot stop from {self.get_state_string()} state")
                print('Machine must be in COOLING or transitional state')

        elif command == 'preheat':
            temp = int(json_data.get('value', 0))
            if 20 <= temp <= 350:
                print(f"Setting preheat temperature to {temp}°C")
                self.set_preheat(temp)
            else:
                print('Preheat temperature must be between 20°C and 350°C')

        elif command == 'fan':
            value = int(json_data.get('value', 0))
            if 1 <= value <= 12:
                print(f"Setting fan to {value}")
                self.set_fan(value)
            else:
                print('Fan value must be between 1 and 12')

        elif command == 'heater':
            value = int(json_data.get('value', 0))
            if 0 <= value <= 14:
                print(f"Setting heater to {value}")
                self.set_heater(value)
            else:
                print('Heater value must be between 0 and 14')

        elif command == 'drum':
            value = int(json_data.get('value', 0))
            if 1 <= value <= 9:
                print(f"Setting drum to {value}")
                self.set_drum(value)
            else:
                print('Drum value must be between 1 and 9')

        else:
            print(f"Unknown command: {command}")

    def set_preheat(self, temp: int) -> None:
        """Set preheat temperature (R2 only)"""
        self.__dbg('aillio_rs:set_preheat()')

        cmd = [0x35, 0x00, 0x00, 0x00]
        cmd[3] = temp & 0xFF
        cmd[2] = (temp >> 8) & 0xFF
        cmdOut = self.prepare_command(cmd)
        if self.parent_pipe is not None:
            self.parent_pipe.send(bytes(cmdOut))
            print(f"Sent preheat command: temp={temp}°C")

if __name__ == '__main__':
    R2 = AillioDummy(debug=True)
    R2.set_heater(5)
    try:
        R2._open_port() # pylint: disable=protected-access
        print(f"Connected to {R2.model} using protocol V{R2.protocol}")

        # Example reading loop
        elapse_time = 1
        while True:
            elapse_time += 1
            print(elapse_time)
            print(f"IBTS: {R2.get_bt():.1f}°C (RoR: {R2.get_bt_ror():.1f}°C/min), "
                f"Probe: {R2.get_dt():.1f}°C (RoR: {R2.get_dt_ror():.1f}°C/min), "
                f"Power: {R2.get_heater()}, Fan: {R2.get_fan()}, "
                f"State: {R2.get_state_string()}, "
                f"Hot Air: {R2.exitt:.1f}°C, "
                f"Inlet: {R2.irt:.1f}°C")
            time.sleep(1)
            R2.tick()
    except KeyboardInterrupt:
        print('\nExiting...')
    except OSError as e:
        print(f"Error: {e}")
    finally:
        R2._close_port() # pylint: disable=protected-access
