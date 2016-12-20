#include <stdio.h>
#include "windows.h"

BOOL APIENTRY DllMain(HMODULE hModule,
	DWORD  ul_reason_for_call,
	LPVOID lpReserved
	)
{
	switch (ul_reason_for_call)
	{
	case DLL_PROCESS_ATTACH:
	case DLL_THREAD_ATTACH:
	case DLL_THREAD_DETACH:
	case DLL_PROCESS_DETACH:
		break;
	}
	return TRUE;
}

void Log(wchar_t *szFormat, ...){
#define NUMCHARS	1024
	wchar_t buf[NUMCHARS];  // Large buffer for long filenames or URLs
	const int LASTCHAR = NUMCHARS - 1;

	// Format the input string
	va_list pArgs;
	va_start(pArgs, szFormat);

	// Use a bounded buffer size to prevent buffer overruns.  Limit count to
	// character size minus one to allow for a NULL terminating character.
	_vsnwprintf_s(buf, NUMCHARS - 1, szFormat, pArgs);
	va_end(pArgs);

	// Ensure that the formatted string is NULL-terminated
	buf[LASTCHAR] = L'\0';
	wcscat_s(buf, L"\r\n");

	OutputDebugStringW(buf);
}

void LogA(char *szFormat, ...){
#define NUMCHARS	1024
	char buf[NUMCHARS];  // Large buffer for long filenames or URLs
	const int LASTCHAR = NUMCHARS - 1;

	// Format the input string
	va_list pArgs;
	va_start(pArgs, szFormat);

	// Use a bounded buffer size to prevent buffer overruns.  Limit count to
	// character size minus one to allow for a NULL terminating character.
	_vsnprintf_s(buf, NUMCHARS - 1, szFormat, pArgs);
	va_end(pArgs);

	// Ensure that the formatted string is NULL-terminated
	buf[LASTCHAR] = '\0';
	strcat_s(buf, "\r\n");

	OutputDebugString(buf);
}