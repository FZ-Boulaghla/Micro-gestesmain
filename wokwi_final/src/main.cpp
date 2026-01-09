
#include <Arduino.h>
#include <SPI.h>
#include <Wire.h>
#include <math.h>
#include <Adafruit_GFX.h>
#include <Adafruit_ILI9341.h>
#include <Adafruit_FT6206.h>

// ===== TFLite Micro =====
#include "model_data.h"  // ton modèle int8: model_int8_tflite, model_int8_tflite_len
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// ---------- Pins (doivent correspondre au diagram.json) ----------
#define TFT_DC   2
#define TFT_CS   15
#define TFT_RST  4   // utilisé par le board-ili9341-cap-touch
#define TFT_LED  6   // rétroéclairage (optionnel)
#define I2C_SDA 10   // FT6206 SDA
#define I2C_SCL  8   // FT6206 SCL

#define LED_RED_PIN     47
#define LED_YELLOW_PIN  48
#define LED_GREEN_PIN   14

// ---------- Objets ----------
Adafruit_ILI9341 tft(TFT_CS, TFT_DC, TFT_RST);
Adafruit_FT6206   ts;

// ---------- UI ----------
#define BOXSIZE   40
#define PENRADIUS 3

// ---------- Détection & tampon geste ----------
struct TouchStats {
  bool     touching      = false;
  uint32_t t_start_ms    = 0;
  uint32_t t_last_ms     = 0;
  int      x_start       = 0, y_start = 0;
  int      x_last        = 0, y_last  = 0;
  float    path_len_px   = 0.0f;   // longueur cumulée du trajet (px)
  int      dir_changes   = 0;      // changements de direction (approx pour transitions)
  float    last_angle    = NAN;    // en radians
};

TouchStats stats;

static uint32_t last_gesture_end_ms = 0;   // pour temps_repos_avant_ms
static const uint32_t WINDOW_MS = 2000;    // même fenêtre 2s que le dataset

// ========== Helpers ==========
float dist_px(int x1,int y1,int x2,int y2) {
  float dx = float(x2-x1), dy=float(y2-y1);
  return sqrtf(dx*dx+dy*dy);
}

void ledsOff() {
  digitalWrite(LED_RED_PIN,    LOW);
  digitalWrite(LED_YELLOW_PIN, LOW);
  digitalWrite(LED_GREEN_PIN,  LOW);
}

void showClass(const char* name, uint16_t color) {
  tft.fillRect(0, 200, 320, 40, ILI9341_BLACK);
  tft.setCursor(8, 210);
  tft.setTextColor(color);
  tft.setTextSize(2);
  tft.print("Classe: ");
  tft.print(name);
}

// ========== TFLite Micro ==========
constexpr int kArenaSize = 25 * 1024;
alignas(16) uint8_t tensor_arena[kArenaSize];

tflite::MicroInterpreter* interpreter = nullptr;
const tflite::Model* model = nullptr;
tflite::AllOpsResolver resolver;            // simple et suffit pour MLP
tflite::MicroErrorReporter micro_error_reporter;

TfLiteTensor* input  = nullptr;
TfLiteTensor* output = nullptr;

// === Normalisation MIN-MAX (extraites de ton dataset) ===
// ratio_occupation, nb_transitions, duree_max_bloc_1_ms, duree_activation_estimee_ms,
// temps_repos_avant_ms, temps_repos_apres_ms
const int NUM_FEATS = 6;
float FEAT_MIN[NUM_FEATS] = {0.015f, 2.0f, 30.0f, 30.0f, 32.5f, 5.0f};
float FEAT_MAX[NUM_FEATS] = {0.74875f, 4.0f, 1497.5f, 1497.5f, 1217.5f, 1225.0f};

void scale_feats(const float in[NUM_FEATS], float out[NUM_FEATS]) {
  for (int i = 0; i < NUM_FEATS; i++) {
    float d = FEAT_MAX[i] - FEAT_MIN[i];
    if (d < 1e-6f) d = 1e-6f;
    float v = (in[i] - FEAT_MIN[i]) / d;
    if (v < 0) v = 0;
    if (v > 1) v = 1;
    out[i] = v;
  }
}

// === Reconstruction EXACTE des 6 features pour TON modèle ===
// D'après le dataset :
//  0) ratio_occupation              ~ durée_geste / (fenêtre 2000 ms)
//  1) nb_transitions                2 ou 4 (approx. via changements de direction)
//  2) duree_max_bloc_1_ms           = durée_geste (contact continu)
//  3) duree_activation_estimee_ms   = durée_geste
//  4) temps_repos_avant_ms          = temps depuis fin du dernier geste
//  5) temps_repos_apres_ms          = 0 en temps réel (on classe à la fin du geste)
void compute_features_tflm(float feats[NUM_FEATS],
                           uint32_t dur_ms,
                           float path_len_px,
                           float speed_px_per_ms,
                           uint32_t repos_avant_ms,
                           int dir_changes)
{
  // 0) ratio_occupation
  float ratio_occupation = float(dur_ms) / float(WINDOW_MS);
  if (ratio_occupation > 1.0f) ratio_occupation = 1.0f;

  // 1) nb_transitions (2 ou 4)
  // seuil simple : si le trajet change de direction "souvent" -> 4 sinon 2
  int nb_transitions = (dir_changes >= 2) ? 4 : 2;

  // 2) & 3) durées
  float duree_max_bloc_1_ms         = float(dur_ms);
  float duree_activation_estimee_ms = float(dur_ms);

  // 4) & 5) repos
  float temps_repos_avant_ms = float(repos_avant_ms);
  float temps_repos_apres_ms = 0.0f;

  feats[0] = ratio_occupation;
  feats[1] = float(nb_transitions);
  feats[2] = duree_max_bloc_1_ms;
  feats[3] = duree_activation_estimee_ms;
  feats[4] = temps_repos_avant_ms;
  feats[5] = temps_repos_apres_ms;
}

// ========== Setup ==========
void setup() {
  Serial.begin(115200);

  pinMode(LED_RED_PIN,    OUTPUT);
  pinMode(LED_YELLOW_PIN, OUTPUT);
  pinMode(LED_GREEN_PIN,  OUTPUT);
  ledsOff();

  // Rétroéclairage (optionnel)
  pinMode(TFT_LED, OUTPUT);
  digitalWrite(TFT_LED, HIGH);

  // TFT
  tft.begin();
  tft.setRotation(4);
  tft.fillScreen(ILI9341_BLACK);
  
  // I²C + FT6206
  Wire.begin(I2C_SDA, I2C_SCL);
  if (!ts.begin(40)) {
    Serial.println("FT6206 init FAIL");
    while (1) { delay(10); }
  }

  // UI: bandeau de couleurs démo
  tft.fillRect(0,            0, BOXSIZE, BOXSIZE, ILI9341_RED);
  tft.fillRect(BOXSIZE,      0, BOXSIZE, BOXSIZE, ILI9341_YELLOW);
  tft.fillRect(BOXSIZE * 2,  0, BOXSIZE, BOXSIZE, ILI9341_GREEN);
  tft.fillRect(BOXSIZE * 3,  0, BOXSIZE, BOXSIZE, ILI9341_CYAN);
  tft.fillRect(BOXSIZE * 4,  0, BOXSIZE, BOXSIZE, ILI9341_BLUE);
  tft.fillRect(BOXSIZE * 5,  0, BOXSIZE, BOXSIZE, ILI9341_MAGENTA);
  tft.drawRect(0, 0, BOXSIZE, BOXSIZE, ILI9341_WHITE);

  tft.setCursor(8, 60);
  tft.setTextColor(ILI9341_WHITE);
  tft.setTextSize(2);
  tft.println("Tap / Swipe detection (ESP32-S3) + ML.");

  // ====== TFLite Micro Init ======
  model = tflite::GetModel(model_int8_tflite);
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kArenaSize, &micro_error_reporter);

  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("TFLM AllocateTensors FAILED");
    while (1) delay(100);
  }

  input  = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("TFLM Loaded OK");
}

// ========== Loop ==========
void loop() {
  delay(10);

  bool isTouched = ts.touched();
  uint32_t now_ms = millis();

  if (!isTouched) {
    // Fin de contact : décider du geste
    if (stats.touching) {
      stats.touching = false;

      uint32_t dur_ms = (stats.t_last_ms - stats.t_start_ms);
      float    D      = dist_px(stats.x_start, stats.y_start, stats.x_last, stats.y_last);
      float    speed  = (dur_ms > 0) ? (D / dur_ms) : 0.0f;

      // repos avant (depuis le dernier geste terminé)
      uint32_t repos_avant_ms = (stats.t_start_ms > last_gesture_end_ms)
                                ? (stats.t_start_ms - last_gesture_end_ms) : 0;

      // log debug
      Serial.printf("Gesture end: dur=%lu ms, dist=%.1f px, speed=%.3f px/ms, path=%.1f px, dir_changes=%d\n",
                    (unsigned long)dur_ms, D, speed, stats.path_len_px, stats.dir_changes);

      // === Préparation features ML ===
      float feats[NUM_FEATS];
      compute_features_tflm(feats, dur_ms, stats.path_len_px, speed, repos_avant_ms, stats.dir_changes);

      // Normalisation min-max
      float feats_n[NUM_FEATS];
      scale_feats(feats, feats_n);

      // Copier dans le tenseur (int8 quantisé)
      for (int i = 0; i < NUM_FEATS; i++) {
        float scaled = feats_n[i];
        // v_q = v_f / scale + zero_point
        int32_t q = (int32_t)roundf(scaled / input->params.scale + input->params.zero_point);
        if (q < -128) q = -128;
        if (q >  127) q =  127;
        input->data.int8[i] = (int8_t)q;
      }

      // Inference
      if (interpreter->Invoke() != kTfLiteOk) {
        Serial.println("Invoke FAILED");
        return;
      }

      // Output (3 classes int8)
      int8_t o0 = output->data.int8[0];  // tap
      int8_t o1 = output->data.int8[1];  // swipe_rapide
      int8_t o2 = output->data.int8[2];  // swipe_lent

      // Argmax
      int cls = 0;
      int8_t maxv = o0;
      if (o1 > maxv) { cls = 1; maxv = o1; }
      if (o2 > maxv) { cls = 2; maxv = o2; }

      ledsOff();
      if (cls == 0) {
        digitalWrite(LED_RED_PIN, HIGH);
        showClass("tap", ILI9341_RED);
      } else if (cls == 1) {
        digitalWrite(LED_YELLOW_PIN, HIGH);
        showClass("swipe_rapide", ILI9341_YELLOW);
      } else {
        digitalWrite(LED_GREEN_PIN, HIGH);
        showClass("swipe_lent", ILI9341_GREEN);
      }

      Serial.printf("ML CLASS = %d (tap=0, swipe_rapide=1, swipe_lent=2)\n", cls);

      // mettre à jour le marqueur de fin de geste pour prochain repos_avant
      last_gesture_end_ms = now_ms;

      // reset stats
      stats.path_len_px = 0.0f;
      stats.dir_changes = 0;
      stats.last_angle  = NAN;
    }
    return;
  }

  // Contact en cours : récupérer point
 
TS_Point p = ts.getPoint();
int16_t tx = p.x, ty = p.y;
p.x = 320 - ty;   // x <- 320 - y
p.y = tx;         // y <- x


  // Première frame du contact
  if (!stats.touching) {
    stats.touching   = true;
    stats.t_start_ms = now_ms;
    stats.x_start    = p.x;
    stats.y_start    = p.y;
    stats.x_last     = p.x;
    stats.y_last     = p.y;
    stats.path_len_px = 0.0f;
    stats.dir_changes = 0;
    stats.last_angle  = NAN;
  }

  // Mise à jour du chemin (longueur + changements de direction)
  {
    float step = dist_px(stats.x_last, stats.y_last, p.x, p.y);
    stats.path_len_px += step;

    // angle de déplacement
    if (step > 0.5f) {
      float angle = atan2f(float(p.y - stats.y_last), float(p.x - stats.x_last)); // [-pi, pi]
      if (!isnan(stats.last_angle)) {
        float dtheta = fabsf(angle - stats.last_angle);
        // normaliser angle diff à [0, pi]
        while (dtheta > PI) dtheta -= 2.0f * PI;
        dtheta = fabsf(dtheta);
        // si changement > ~45° on incrémente
        if (dtheta > (PI / 4.0f)) {
          stats.dir_changes++;
        }
      }
      stats.last_angle = angle;
    }
  }

  stats.x_last    = p.x;
  stats.y_last    = p.y;
  stats.t_last_ms = now_ms;

  // Dessin (évite de peindre sur la barre d'outils en haut)
  if ((p.y - PENRADIUS) > BOXSIZE && (p.y + PENRADIUS) < tft.height()) {
    tft.fillCircle(p.x, p.y, PENRADIUS, ILI9341_WHITE);
  }
}
