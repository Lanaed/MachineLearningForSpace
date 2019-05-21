#include "libschsat.h"

void control(void) {
	int16_t  v1, v2, v3;
	uint16_t num = 1, x, y;
	int i;
	
	sun_sensor_turn_on(1);
	sun_sensor_turn_on(2);
	sun_sensor_turn_on(3);
	sun_sensor_turn_on(4);
	hyro_turn_on(num);
	
	Sleep(1);
	
	for (i=0; i<5000; i++) {
		if (LSS_OK == sun_sensor_request_raw (1, &x, &y)) {
			printf("%d;%d;", x, y);
		}
		else printf("er1");
		
		if (LSS_OK == sun_sensor_request_raw (2, &x, &y)) {
			printf("%d;%d;", x, y);
		}
		else printf("er2");
		
		if (LSS_OK == sun_sensor_request_raw (3, &x, &y)) {
			printf("%d;%d;", x, y);
		}
		else printf("er3");
		
		if (LSS_OK == sun_sensor_request_raw (4, &x, &y)) {
			printf("%d;%d;", x, y);
		}
		else printf("er4");
		
		if (LSS_OK == hyro_request_raw(num, &v1, &v2, &v3)) {
			printf("%d;%d;%d", v1, v2, v3);
		}
		else printf("dus");
		
		printf("\n");
		Sleep(0.05);
	}
	
	sun_sensor_turn_off(1);
	sun_sensor_turn_off(2);
	sun_sensor_turn_off(3);
	sun_sensor_turn_off(4);
	hyro_turn_off(num);
}