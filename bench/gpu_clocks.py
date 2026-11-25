"""GPU clocks/power telemetry through the AMD Display Library (ADL).

ZLUDA NVML reports NotSupported for clocks and temperature, and the ROCm 10.1
Windows user space ships no rocm-smi/amd-smi. The Adrenalin driver does install
`atiadlxx.dll`, which exposes engine/memory clocks, temperature, fan and board
power through the legacy Overdrive5/Overdrive6 interfaces.

This module wraps the few ADL calls needed for reproducible benchmarks. It
degrades gracefully: if ADL is missing, no AMD adapter is present, or the
driver rejects the struct sizes, `sample()` returns None values with an error
string instead of raising.

Legacy Overdrive5 calls return ADL_ERR on current Adrenalin drivers, so clocks
come from ADL2_Overdrive6_CurrentStatus_Get (needs the ADL2 control handle).

Usage:
  venv\\Scripts\\python.exe bench\\gpu_clocks.py            # one sample
  venv\\Scripts\\python.exe bench\\gpu_clocks.py --seconds 5 --interval 0.5
  venv\\Scripts\\python.exe bench\\gpu_clocks.py --json     # machine readable

Clocks are reported in MHz (ADL uses 10 kHz units internally); temperature in
degrees Celsius (ADL uses 1/1000 C); power in percent of the board limit.
"""

import argparse
import ctypes
import json
import sys
import time
from ctypes import CFUNCTYPE, POINTER, Structure, byref, c_char, c_int, c_uint, c_void_p

ADL_MAX_PATH = 256
ADL_OK = 0

# ADL_Overdrive6_CurrentPower_Get state info selector (percent of board power).
ADL_OD6_GETSTATEINFO = 0x00000001


class AdapterInfo(Structure):
    _fields_ = [
        ("iSize", c_int),
        ("iAdapterIndex", c_int),
        ("strUDID", c_char * ADL_MAX_PATH),
        ("iBusNumber", c_int),
        ("iDeviceNumber", c_int),
        ("iFunctionNumber", c_int),
        ("iVendorID", c_int),
        ("strAdapterName", c_char * ADL_MAX_PATH),
        ("strDisplayName", c_char * ADL_MAX_PATH),
        ("iPresent", c_int),
        ("iExist", c_int),
        ("strDriverPath", c_char * ADL_MAX_PATH),
        ("strDriverPathExt", c_char * ADL_MAX_PATH),
        ("strPNPString", c_char * ADL_MAX_PATH),
        ("iOSDisplayIndex", c_int),
    ]


class ADLODCurrentActivity(Structure):
    _fields_ = [
        ("iAdapterIndex", c_int),
        ("iEngineClock", c_int),
        ("iMemoryClock", c_int),
        ("iVddc", c_int),
        ("iTemperature", c_int),
        ("iActivity", c_int),
        ("iCurrentPerformanceLevel", c_int),
        ("iCurrentBusSpeed", c_int),
        ("iCurrentBusLanes", c_int),
        ("iMaximumBusLanes", c_int),
        ("iReserved", c_int * 4),
    ]


class ADLOD6CurrentStatus(Structure):
    """ADL2_Overdrive6_CurrentStatus_Get result: clocks in 10 kHz units."""

    _fields_ = [
        ("iEngineClock", c_int),
        ("iMemoryClock", c_int),
        ("iActivity", c_int),
    ]


class ADLODNPerformanceStatus(Structure):
    """ADL2_OverdriveN_PerformanceStatus_Get result (RDNA-era live clocks)."""

    _fields_ = [
        ("iCoreClock", c_int),
        ("iMemoryClock", c_int),
        ("iDCEFClock", c_int),
        ("iGFXClock", c_int),
        ("iUVDClock", c_int),
        ("iVCEClock", c_int),
        ("iGPUActivityPercent", c_int),
        ("iCurrentCorePerformanceLevel", c_int),
        ("iCurrentMemoryPerformanceLevel", c_int),
        ("iCurrentDCEFPerformanceLevel", c_int),
        ("iCurrentGFXPerformanceLevel", c_int),
        ("iUVDPerformanceLevel", c_int),
        ("iUVDEfficiency", c_int),
        ("iVCEPerformanceLevel", c_int),
        ("iVCEEfficiency", c_int),
        ("iVFCEfficiency", c_int),
        ("iReserved", c_int * 16),
    ]


class ActivitySample(Structure):
    """Reserved for future Overdrive6 activity sampling."""

    _fields_ = [
        ("iAdapterIndex", c_int),
        ("iActivity", c_int),
    ]


ADL_MAIN_MALLOC_CALLBACK = CFUNCTYPE(c_void_p, c_int)

_alloc_keepalive = []


def _malloc(size):
    # ADL keeps the pointer; hold the buffer for the process lifetime.
    buf = ctypes.create_string_buffer(max(size, 1))
    _alloc_keepalive.append(buf)
    return ctypes.cast(buf, c_void_p)


class ADL:
    def __init__(self):
        self.lib = ctypes.CDLL("atiadlxx.dll")
        self.callback = ADL_MAIN_MALLOC_CALLBACK(_malloc)
        self.control = c_void_p()
        self.error = None

        self.lib.ADL_Main_Control_Create.argtypes = [ADL_MAIN_MALLOC_CALLBACK, c_int]
        self.lib.ADL_Main_Control_Create.restype = c_int
        self.lib.ADL_Main_Control_Destroy.argtypes = []
        self.lib.ADL_Main_Control_Destroy.restype = c_int
        self.lib.ADL_Adapter_NumberOfAdapters_Get.argtypes = [POINTER(c_int)]
        self.lib.ADL_Adapter_NumberOfAdapters_Get.restype = c_int
        self.lib.ADL_Adapter_AdapterInfo_Get.argtypes = [POINTER(AdapterInfo), c_int]
        self.lib.ADL_Adapter_AdapterInfo_Get.restype = c_int
        self.lib.ADL_Adapter_Active_Get.argtypes = [c_int, POINTER(c_int)]
        self.lib.ADL_Adapter_Active_Get.restype = c_int
        self.lib.ADL_Overdrive5_CurrentActivity_Get.argtypes = [
            c_int, POINTER(ADLODCurrentActivity)
        ]
        self.lib.ADL_Overdrive5_CurrentActivity_Get.restype = c_int
        self.lib.ADL2_Overdrive6_CurrentPower_Get.argtypes = [
            c_void_p, c_int, c_uint, POINTER(c_int)
        ]
        self.lib.ADL2_Overdrive6_CurrentPower_Get.restype = c_int
        self.lib.ADL2_Main_Control_Create.argtypes = [
            ADL_MAIN_MALLOC_CALLBACK, c_int, POINTER(c_void_p)
        ]
        self.lib.ADL2_Main_Control_Create.restype = c_int
        self.lib.ADL2_Main_Control_Destroy.argtypes = [c_void_p]
        self.lib.ADL2_Overdrive6_CurrentStatus_Get.argtypes = [
            c_void_p, c_int, POINTER(ADLOD6CurrentStatus)
        ]
        self.lib.ADL2_Overdrive6_CurrentStatus_Get.restype = c_int
        self.lib.ADL2_Overdrive6_Temperature_Get.argtypes = [
            c_void_p, c_int, POINTER(c_int)
        ]
        self.lib.ADL2_Overdrive6_Temperature_Get.restype = c_int
        self.lib.ADL2_OverdriveN_PerformanceStatus_Get.argtypes = [
            c_void_p, c_int, POINTER(ADLODNPerformanceStatus)
        ]
        self.lib.ADL2_OverdriveN_PerformanceStatus_Get.restype = c_int
        self.control2 = c_void_p()
        self.error2 = None

        status = self.lib.ADL_Main_Control_Create(self.callback, 1)
        if status != ADL_OK:
            self.error = f"ADL_Main_Control_Create failed: {status}"
            return

        count = c_int(0)
        status = self.lib.ADL_Adapter_NumberOfAdapters_Get(byref(count))
        if status != ADL_OK or count.value <= 0:
            self.error = f"ADL_Adapter_NumberOfAdapters_Get failed: {status}"
            return

        infos = (AdapterInfo * count.value)()
        for info in infos:
            info.iSize = ctypes.sizeof(AdapterInfo)
        status = self.lib.ADL_Adapter_AdapterInfo_Get(
            infos, ctypes.sizeof(AdapterInfo) * count.value
        )
        if status != ADL_OK:
            self.error = f"ADL_Adapter_AdapterInfo_Get failed: {status}"
            return

        self.adapters = []
        for info in infos:
            if not info.iPresent:
                continue
            active = c_int(0)
            if self.lib.ADL_Adapter_Active_Get(info.iAdapterIndex, byref(active)) != ADL_OK:
                continue
            if not active.value:
                continue
            name = info.strAdapterName.decode(errors="replace") or info.strPNPString.decode(
                errors="replace"
            )
            self.adapters.append((info.iAdapterIndex, name))

        status = self.lib.ADL2_Main_Control_Create(
            self.callback, 1, byref(self.control2)
        )
        if status != ADL_OK:
            self.error2 = f"ADL2_Main_Control_Create failed: {status}"

    def sample(self):
        result = {
            "adapter": None,
            "engine_clock_mhz": None,
            "memory_clock_mhz": None,
            "vddc_mv": None,
            "temperature_c": None,
            "activity_pct": None,
            "power_pct": None,
            "errors": {},
        }
        if self.error:
            result["errors"]["adl"] = self.error
            return result
        if not self.adapters:
            result["errors"]["adl"] = "no active AMD adapter"
            return result
        index, name = self.adapters[0]
        result["adapter"] = name

        if not self.error2:
            # Prefer the OverdriveN status API: RDNA cards report live clocks
            # there while Overdrive5/6 return ADL_ERR_NOT_SUPPORTED.
            odnn = ADLODNPerformanceStatus()
            status = self.lib.ADL2_OverdriveN_PerformanceStatus_Get(
                self.control2, index, byref(odnn)
            )
            if status == ADL_OK:
                result["engine_clock_mhz"] = round(odnn.iCoreClock / 100.0, 1)
                result["memory_clock_mhz"] = round(odnn.iMemoryClock / 100.0, 1)
                result["activity_pct"] = odnn.iGPUActivityPercent
                result["clock_source"] = "overdriveN"
            else:
                result["errors"]["overdriveN_status"] = status

            if result["engine_clock_mhz"] is None:
                status_adl = ADLOD6CurrentStatus()
                status = self.lib.ADL2_Overdrive6_CurrentStatus_Get(
                    self.control2, index, byref(status_adl)
                )
                if status == ADL_OK:
                    result["engine_clock_mhz"] = round(status_adl.iEngineClock / 100.0, 1)
                    result["memory_clock_mhz"] = round(status_adl.iMemoryClock / 100.0, 1)
                    result["activity_pct"] = status_adl.iActivity
                    result["clock_source"] = "overdrive6"
                else:
                    result["errors"]["overdrive6_status"] = status

            temperature = c_int(0)
            status = self.lib.ADL2_Overdrive6_Temperature_Get(
                self.control2, index, byref(temperature)
            )
            if status == ADL_OK:
                result["temperature_c"] = round(temperature.value / 1000.0, 1)
            else:
                result["errors"]["overdrive6_temperature"] = status

            power = c_int(0)
            status = self.lib.ADL2_Overdrive6_CurrentPower_Get(
                self.control2, index, ADL_OD6_GETSTATEINFO, byref(power)
            )
            if status == ADL_OK:
                result["power_pct"] = power.value
            else:
                result["errors"]["overdrive6_power"] = status
        else:
            result["errors"]["adl2"] = self.error2
        return result

    def close(self):
        if not self.error:
            self.lib.ADL_Main_Control_Destroy()
        if not self.error2:
            self.lib.ADL2_Main_Control_Destroy(self.control2)


def sample():
    try:
        adl = ADL()
    except OSError as exc:
        return {"errors": {"adl": f"cannot load atiadlxx.dll: {exc}"}}
    try:
        return adl.sample()
    finally:
        adl.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=0.0)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.seconds <= 0:
        result = sample()
        print(json.dumps(result, indent=2) if args.json else result)
        return 0 if not result.get("errors", {}).get("adl") else 1

    deadline = time.perf_counter() + args.seconds
    samples = []
    while time.perf_counter() < deadline:
        samples.append(sample())
        time.sleep(args.interval)
    valid = [s for s in samples if s.get("engine_clock_mhz")]
    summary = {
        "samples": len(samples),
        "engine_clock_mhz": {
            "min": min((s["engine_clock_mhz"] for s in valid), default=None),
            "max": max((s["engine_clock_mhz"] for s in valid), default=None),
            "avg": round(sum(s["engine_clock_mhz"] for s in valid) / len(valid), 1)
            if valid
            else None,
        },
        "temperature_c": {
            "min": min((s["temperature_c"] for s in valid), default=None),
            "max": max((s["temperature_c"] for s in valid), default=None),
        },
        "raw": samples if args.json else None,
        "errors": samples[0].get("errors", {}) if samples else {},
    }
    print(json.dumps(summary, indent=2) if args.json else summary)
    return 0 if valid else 1


if __name__ == "__main__":
    sys.exit(main())
