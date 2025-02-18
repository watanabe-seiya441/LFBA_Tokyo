#include <Arduino.h>

const int ledPins[] = {2, 3, 4};
const int relayPins[] = {6, 7, 8, 9};
const int buttonPins[] = {A0, A1, A2, A3, 5};
const int toggleSwitches[] = {A4, A5};

uint8_t state = 0;

void setup() {
    Serial.begin(9600);
    for (int pin : ledPins) pinMode(pin, OUTPUT);
    for (int pin : relayPins) pinMode(pin, OUTPUT);
    for (int pin : buttonPins) pinMode(pin, INPUT_PULLUP);
    for (int pin : toggleSwitches) pinMode(pin, INPUT_PULLUP);

    // Initialize toggle switch states before sending notification
    state |= (digitalRead(toggleSwitches[0]) << 5) | (digitalRead(toggleSwitches[1]) << 6);

    sendNotification('S');
}

void loop() {
    // Handle button presses
    bool is_auto = state & (1 << 5);
    if (!is_auto) {
        for (int i = 0; i < 5; i++) {
            if (digitalRead(buttonPins[i]) == LOW) {
                while (digitalRead(buttonPins[i]) == LOW); // Debounce
                state ^= (1 << i);
                sendNotification('S');
            }
        }
    }
    
    // Update toggle switch states
    state = (state & 0b00011111) | (digitalRead(toggleSwitches[0]) << 5) | (digitalRead(toggleSwitches[1]) << 6);

    // Handle serial input when in manual mode
    if (Serial.available()) {
        String received = Serial.readStringUntil('\n');
        if (is_auto && received.startsWith("C") && received.length() == 5) {
            for (int i = 0; i < 4; i++) {
                state = (state & ~(1 << i)) | ((received[i + 1] - '0') << i);
            }
        }
    }

    // Control relay outputs
    for (int i = 0; i < 4; i++) digitalWrite(relayPins[i], state & (1 << i));
    
    // Control LED indicators
    digitalWrite(ledPins[0], state & (1 << 5));
    digitalWrite(ledPins[1], state & (1 << 6));
    digitalWrite(ledPins[2], !(state & (1 << 6)));

    delay(100);
}

// Sends notification to Serial monitor
void sendNotification(char type) {
    Serial.print(type);
    for (int i = 0; i < 4; i++) Serial.print((state & (1 << i)) ? "1" : "0");
    Serial.print((state & (1 << 5)) ? "1" : "0");
    Serial.print((state & (1 << 6)) ? "1" : "0");
    Serial.println();
}