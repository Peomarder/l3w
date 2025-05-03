#include <iostream>
#include <windows.h>
#include <psapi.h>
#include <vector>
#include <string>
#include <ctime>
#include <stdio.h>
#include <tlhelp32.h>
#include <comdef.h>
#include <Wbemidl.h>
#include <pdh.h>

#pragma comment(lib, "pdh.lib")
#pragma comment(lib, "psapi.lib")
#pragma comment(lib, "wbemuuid.lib")

// Structure to hold process information
struct ProcessInfo {
    DWORD pid;
    std::string name;
};

// Function to display error message for COM errors
void DisplayErrorMessage(HRESULT hResult) {
    _com_error error(hResult);
    std::wcerr << L"Error: " << error.ErrorMessage() << std::endl;
}

// Function to list all running processes
std::vector<ProcessInfo> listProcesses() {
    std::vector<ProcessInfo> processes;
    HANDLE hProcessSnap;
    PROCESSENTRY32 pe32;

    // Take a snapshot of all processes in the system
    hProcessSnap = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    if (hProcessSnap == INVALID_HANDLE_VALUE) {
        std::cerr << "CreateToolhelp32Snapshot failed with error " << GetLastError() << std::endl;
        return processes;
    }

    pe32.dwSize = sizeof(PROCESSENTRY32);

    // Retrieve information about the first process
    if (!Process32First(hProcessSnap, &pe32)) {
        std::cerr << "Process32First failed with error " << GetLastError() << std::endl;
        CloseHandle(hProcessSnap);
        return processes;
    }

    // Now walk the snapshot of processes
    do {
        ProcessInfo pi;
        pi.pid = pe32.th32ProcessID;
        pi.name = pe32.szExeFile;
        processes.push_back(pi);
    } while (Process32Next(hProcessSnap, &pe32));

    CloseHandle(hProcessSnap);
    return processes;
}

// Function to check if a process is still running
bool isProcessRunning(DWORD pid) {
    HANDLE process = OpenProcess(SYNCHRONIZE, FALSE, pid);
    if (process == NULL) {
        return false;
    }
    DWORD ret = WaitForSingleObject(process, 0);
    CloseHandle(process);
    return (ret == WAIT_TIMEOUT);
}

// Structure to hold performance data
struct PerformanceData {
    ULONGLONG pageFaults;
    ULONGLONG hardFaults;
    ULONGLONG workingSet;
    ULONGLONG pageFileUsage;
};

// Function to get process performance data directly using PSAPI
bool getProcessPerformanceData(DWORD pid, PerformanceData& data) {
    HANDLE hProcess = OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, FALSE, pid);
    if (hProcess == NULL) {
        std::cerr << "Failed to open process (PID: " << pid << ") with error " << GetLastError() << std::endl;
        return false;
    }

    PROCESS_MEMORY_COUNTERS_EX pmc;
    if (GetProcessMemoryInfo(hProcess, (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc))) {
        data.pageFaults = pmc.PageFaultCount;
        data.workingSet = pmc.WorkingSetSize;
        data.pageFileUsage = pmc.PagefileUsage;
        
        // Hard faults aren't directly available, but we can estimate them
        // using WMI (Windows Management Instrumentation)
        CloseHandle(hProcess);
        return true;
    }

    CloseHandle(hProcess);
    std::cerr << "GetProcessMemoryInfo failed with error " << GetLastError() << std::endl;
    return false;
}

// Get hard page faults using WMI
bool getHardPageFaults(DWORD pid, ULONGLONG& hardFaults) {
    HRESULT hres;
    
    // Initialize COM
    hres = CoInitializeEx(0, COINIT_MULTITHREADED);
    if (FAILED(hres)) {
        std::cerr << "Failed to initialize COM library" << std::endl;
        return false;
    }

    // Initialize security
    hres = CoInitializeSecurity(
        NULL,                       // Security descriptor
        -1,                         // COM authentication
        NULL,                       // Authentication services
        NULL,                       // Reserved
        RPC_C_AUTHN_LEVEL_DEFAULT,  // Default authentication
        RPC_C_IMP_LEVEL_IMPERSONATE,// Default impersonation
        NULL,                       // Authentication info
        EOAC_NONE,                  // Additional capabilities
        NULL                        // Reserved
    );

    if (FAILED(hres)) {
        CoUninitialize();
        std::cerr << "Failed to initialize security. Error code = 0x" 
                  << std::hex << hres << std::dec << std::endl;
        return false;
    }

    // Obtain the WMI locator
    IWbemLocator* pLoc = NULL;
    hres = CoCreateInstance(
        CLSID_WbemLocator,
        0,
        CLSCTX_INPROC_SERVER,
        IID_IWbemLocator,
        (LPVOID*)&pLoc
    );

    if (FAILED(hres)) {
        CoUninitialize();
        std::cerr << "Failed to create IWbemLocator object. Error code = 0x" 
                  << std::hex << hres << std::dec << std::endl;
        return false;
    }

    // Connect to WMI through the IWbemLocator interface
    IWbemServices* pSvc = NULL;
    hres = pLoc->ConnectServer(
        _bstr_t(L"ROOT\\CIMV2"),  // Object path of WMI namespace
        NULL,                      // User name
        NULL,                      // User password
        0,                         // Locale
        0,                      // Security flags
        0,                         // Authority
        0,                         // Context object
        &pSvc                      // IWbemServices proxy
    );

    if (FAILED(hres)) {
        pLoc->Release();
        CoUninitialize();
        std::cerr << "Could not connect to WMI namespace. Error code = 0x" 
                  << std::hex << hres << std::dec << std::endl;
        return false;
    }

    // Set security levels on the proxy
    hres = CoSetProxyBlanket(
        pSvc,                        // Indicates the proxy to set
        RPC_C_AUTHN_WINNT,           // RPC_C_AUTHN_xxx
        RPC_C_AUTHZ_NONE,            // RPC_C_AUTHZ_xxx
        NULL,                        // Server principal name
        RPC_C_AUTHN_LEVEL_CALL,      // RPC_C_AUTHN_LEVEL_xxx
        RPC_C_IMP_LEVEL_IMPERSONATE, // RPC_C_IMP_LEVEL_xxx
        NULL,                        // client identity
        EOAC_NONE                    // proxy capabilities
    );

    if (FAILED(hres)) {
        pSvc->Release();
        pLoc->Release();
        CoUninitialize();
        std::cerr << "Could not set proxy blanket. Error code = 0x" 
                  << std::hex << hres << std::dec << std::endl;
        return false;
    }

    // Use the IWbemServices pointer to make WMI requests
    // Create WQL query for process with specific PID
    wchar_t query[256];
    swprintf_s(query, 256, L"SELECT * FROM Win32_PerfFormattedData_PerfProc_Process WHERE IDProcess=%u", pid);

    IEnumWbemClassObject* pEnumerator = NULL;
    hres = pSvc->ExecQuery(
        bstr_t("WQL"),
        bstr_t(query),
        WBEM_FLAG_FORWARD_ONLY | WBEM_FLAG_RETURN_IMMEDIATELY,
        NULL,
        &pEnumerator
    );

    if (FAILED(hres)) {
        pSvc->Release();
        pLoc->Release();
        CoUninitialize();
        std::cerr << "Query for process performance data failed. Error code = 0x" 
                  << std::hex << hres << std::dec << std::endl;
        return false;
    }

    // Get the data from the query
    hardFaults = 0;
    IWbemClassObject* pclsObj = NULL;
    ULONG uReturn = 0;

    while (pEnumerator) {
        hres = pEnumerator->Next(WBEM_INFINITE, 1, &pclsObj, &uReturn);

        if (0 == uReturn) {
            break;
        }

        VARIANT vtProp;
        VariantInit(&vtProp);

        // Get the value of the PageFaultsPersec property
        hres = pclsObj->Get(L"PageFaultsPersec", 0, &vtProp, 0, 0);
        if (SUCCEEDED(hres)) {
            if (vtProp.vt == VT_I4) {
                hardFaults = vtProp.lVal;
            } else if (vtProp.vt == VT_BSTR) {
                hardFaults = _wtoi(vtProp.bstrVal);
            }
            VariantClear(&vtProp);
        }

        pclsObj->Release();
    }

    // Cleanup
    pEnumerator->Release();
    pSvc->Release();
    pLoc->Release();
    CoUninitialize();

    return true;
}

int main() {
    std::cout << "Enhanced Cache Miss Profiler for Windows 7 Professional SP1\n";
    std::cout << "Listing all running processes...\n\n";

    // Get list of processes
    std::vector<ProcessInfo> processes = listProcesses();

    // Display processes
    std::cout << "Available processes:\n";
    for (size_t i = 0; i < processes.size(); i++) {
        std::cout << i + 1 << ": " << processes[i].name << " (PID: " << processes[i].pid << ")\n";
    }

    // Ask user to select a process
    std::string processName;
    std::cout << "\nEnter the name of the process to profile (e.g., notepad.exe): ";
    std::cin >> processName;

    // Find the process
    DWORD targetPid = 0;
    for (const auto& proc : processes) {
        if (_stricmp(proc.name.c_str(), processName.c_str()) == 0) {
            targetPid = proc.pid;
            break;
        }
    }

    if (targetPid == 0) {
        std::cerr << "Process not found. Exiting...\n";
        return 1;
    }

    std::cout << "Starting profiling for " << processName << " (PID: " << targetPid << ")\n";
    std::cout << "Using direct memory counter access method for Windows 7 Professional SP1\n";

    // Initialize performance data
    PerformanceData initialData = {0};
    if (!getProcessPerformanceData(targetPid, initialData)) {
        std::cerr << "Failed to get initial performance data. Exiting...\n";
        return 1;
    }

    ULONGLONG initialHardFaults = 0;
    if (!getHardPageFaults(targetPid, initialHardFaults)) {
        std::cout << "Warning: Unable to get hard page fault data. Will continue without it.\n";
    }
    
    // Record start time
    time_t startTime = time(NULL);
    
    std::cout << "Profiling... Press Ctrl+C to stop if the application doesn't exit automatically.\n";

    // Monitor the process until it exits
    while (isProcessRunning(targetPid)) {
        // Wait a bit before checking again to reduce CPU usage
        Sleep(500);
    }
    
    // Get final performance data
    PerformanceData finalData = {0};
    if (!getProcessPerformanceData(targetPid, finalData)) {
        std::cerr << "Failed to get final performance data\n";
    }

    ULONGLONG finalHardFaults = 0;
    if (!getHardPageFaults(targetPid, finalHardFaults)) {
        std::cout << "Warning: Unable to get final hard page fault data\n";
    }
    
    // Calculate process runtime
    time_t endTime = time(NULL);
    double runtime = difftime(endTime, startTime);
    
    // Calculate differences
    ULONGLONG pageFaultDelta = finalData.pageFaults - initialData.pageFaults;
    ULONGLONG hardFaultDelta = finalHardFaults - initialHardFaults;
    
    // Display results
    std::cout << "\nProfiling results for " << processName << ":\n";
    std::cout << "Runtime: " << runtime << " seconds\n";
    std::cout << "Total page faults (soft cache misses): " << pageFaultDelta << "\n";
    
    if (finalHardFaults > 0 || initialHardFaults > 0) {
        std::cout << "Total hard page faults (disk reads): " << hardFaultDelta << "\n";
    }
    
    // Estimate cache misses
    // On Windows 7, we can't directly measure CPU cache misses without special drivers,
    // but page faults are a reasonable proxy for memory access inefficiency
    std::cout << "Estimated cache misses (based on page faults): " << pageFaultDelta << "\n";
    
    std::cout << "\nNote: For precise CPU cache miss counting on Windows 7,\n"
              << "consider using Intel's Performance Counter Monitor (PCM) library\n"
              << "or Windows Performance Toolkit with kernel-mode drivers.\n";
    
    return 0;
}
