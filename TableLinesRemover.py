import cv2
import numpy as np
import os
import subprocess
import re
import difflib

# ==========================================
# PART 1: IMAGE CLEANUP & CROPPING
# ==========================================
class TableLinesRemover:
    def __init__(self, image):
        self.image = image.copy()
        self.original = image.copy()

    def execute(self):
        self.isolate_color_blocks()
        self.grayscale_image()
        self.threshold_image()
        self.invert_image()
        self.erode_vertical_lines()
        self.erode_horizontal_lines()
        self.combine_eroded_images()
        self.erase_lines_from_original()
        self.restore_color_blocks()
        self.crop_to_largest_table()
        return self.final_image

    def _make_solid_color_mask(self, hsv, min_area=200):
        raw = cv2.inRange(hsv, np.array([0, 30, 30]), np.array([180, 255, 255]))
        contours, _ = cv2.findContours(raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        solid = np.zeros_like(raw)
        for cnt in contours:
            if cv2.contourArea(cnt) > min_area:
                cv2.drawContours(solid, [cnt], -1, 255, -1)
        return solid

    def isolate_color_blocks(self):
        hsv = cv2.cvtColor(self.original, cv2.COLOR_BGR2HSV)
        self.solid_color_mask = self._make_solid_color_mask(hsv)
        self.isolated_colors = cv2.bitwise_and(self.original, self.original, mask=self.solid_color_mask)

    def grayscale_image(self):
        self.grey = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def threshold_image(self):
        self.thresholded_image = cv2.threshold(self.grey, 150, 255, cv2.THRESH_BINARY)[1]

    def invert_image(self):
        self.inverted_image = cv2.bitwise_not(self.thresholded_image)

    def erode_vertical_lines(self):
        ver = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        self.vertical_lines = cv2.erode(self.inverted_image, ver, iterations=1)
        self.vertical_lines = cv2.dilate(self.vertical_lines, ver, iterations=1)

    def erode_horizontal_lines(self):
        hor = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        self.horizontal_lines = cv2.erode(self.inverted_image, hor, iterations=1)
        self.horizontal_lines = cv2.dilate(self.horizontal_lines, hor, iterations=1)

    def combine_eroded_images(self):
        self.combined_lines = cv2.add(self.vertical_lines, self.horizontal_lines)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.combined_lines = cv2.dilate(self.combined_lines, kernel, iterations=1)

    def erase_lines_from_original(self):
        self.image_without_lines = self.original.copy()
        self.image_without_lines[self.combined_lines == 255] = [255, 255, 255]

    def restore_color_blocks(self):
        inv_color_mask = cv2.bitwise_not(self.solid_color_mask)
        bg_area = cv2.bitwise_and(self.image_without_lines, self.image_without_lines, mask=inv_color_mask)
        self.final_image = cv2.add(bg_area, self.isolated_colors)

    def crop_to_largest_table(self):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        closed_lines = cv2.morphologyEx(self.combined_lines, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        max_area = 0
        best_box = None
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            if area > max_area:
                max_area = area
                best_box = (x, y, w, h)
                
        if best_box is not None:
            x, y, w, h = best_box
            padding = -5
            y_start = max(0, y - padding)
            y_end = min(self.final_image.shape[0], y + h + padding)
            x_start = max(0, x - padding)
            x_end = min(self.final_image.shape[1], x + w + padding)
            self.final_image = self.final_image[y_start:y_end, x_start:x_end]


# ==========================================
# PART 2: OCR & GEOMETRIC DATA EXTRACTION
# ==========================================
class ScheduleDataExtractor:
    def __init__(self, cropped_image):
        self.image = cropped_image
        self.img_h, self.img_w = self.image.shape[:2]

        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        gray_full = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        # 1. DEFINE COLOR FAMILIES FOR BOTH MASKING AND CATEGORIZATION
        self.color_families = [
            # 0: Dark Green (PSM113 / RM319) 
            cv2.inRange(hsv, np.array([35, 40, 20]), np.array([85, 255, 200])),
            # 1: Blue (Administrative Hours)
            cv2.inRange(hsv, np.array([90, 40, 40]), np.array([130, 255, 255])),
            # 2: Red/Peach (PSE105 / 4PSY)
            cv2.bitwise_or(
                cv2.inRange(hsv, np.array([0, 40, 100]), np.array([15, 255, 255])),
                cv2.inRange(hsv, np.array([160, 40, 100]), np.array([180, 255, 255]))
            ),
            # 3: Yellow/Beige (Research / Comm Ext) 
            cv2.inRange(hsv, np.array([15, 20, 150]), np.array([35, 255, 255]))
        ]

        # --- Zone B: Inside Colored Blocks (Per-Color Extraction & Noise Filtering) ---
        self.color_mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
        ink_in_color = np.zeros(self.image.shape[:2], dtype=np.uint8)
        
        for i, fam_mask in enumerate(self.color_families):
            clean_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            fam_mask = cv2.morphologyEx(fam_mask, cv2.MORPH_CLOSE, clean_kernel)
            self.color_mask = cv2.bitwise_or(self.color_mask, fam_mask)

            # Tailor the smoothing and thresholding per color family
            if i == 0:  # Dark Green
                # Target the "blackest" pixels and make them white using a global threshold.
                smooth = cv2.medianBlur(gray_full, 3)
                
                # THRESH_BINARY_INV turns anything darker than a value of 75 into pure white (255).
                # The black text easily falls under 75, while the green background stays above it.
                _, color_text_thresh = cv2.threshold(smooth, 40, 255, cv2.THRESH_BINARY_INV)
                min_area = 5  # We can lower this now since the global threshold creates less static
                min_h = 6
                
            elif i == 1:  # Blue
                smooth = cv2.medianBlur(gray_full, 5)
                smooth = cv2.bilateralFilter(smooth, 9, 75, 75)
                color_text_thresh = cv2.adaptiveThreshold(
                    smooth, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 14
                )
                min_area = 15
                min_h = 6
                
            else:         # Red & Yellow
                smooth = cv2.medianBlur(gray_full, 5)
                color_text_thresh = cv2.adaptiveThreshold(
                    smooth, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 8
                )
                min_area = 12
                min_h = 6

            raw_ink = cv2.bitwise_and(color_text_thresh, color_text_thresh, mask=fam_mask)

            # Filter out all noise using connected components
            family_ink = np.zeros(self.image.shape[:2], dtype=np.uint8)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(raw_ink, connectivity=8)
            
            for lbl in range(1, num_labels):
                area = stats[lbl, cv2.CC_STAT_AREA]
                h = stats[lbl, cv2.CC_STAT_HEIGHT]
                if area >= min_area and h >= min_h:
                    family_ink[labels == lbl] = 255

            # HEALING STEP FOR GREEN
            #if i == 0:
             #   # 1. Swell the strokes
               # repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 3))
               # family_ink = cv2.dilate(family_ink, repair_kernel, iterations=1)
                
                # 2. Fill gaps inside the letters
              #  close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
              #  family_ink = cv2.morphologyEx(family_ink, cv2.MORPH_CLOSE, close_kernel)

            ink_in_color = cv2.bitwise_or(ink_in_color, family_ink)

        # --- Zone A: White background (Outside colored blocks) ---
        _, bw_outside = cv2.threshold(gray_full, 150, 255, cv2.THRESH_BINARY_INV)
        bw_outside_masked = cv2.bitwise_and(bw_outside, bw_outside, mask=cv2.bitwise_not(self.color_mask))
        
        noise_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        bw_opened = cv2.morphologyEx(bw_outside_masked, cv2.MORPH_OPEN, noise_kernel, iterations=1)

        # --- Combine all zones ---
        self.ocr_mask = cv2.bitwise_or(bw_opened, ink_in_color)
        self.bw_image_for_viewing = cv2.bitwise_not(self.ocr_mask)

    def execute(self):
        print("\n[System] Locating schedule boundaries...")
        self.categorize_boxes()
        self.extract_data()
        
    def generate_time_slots(self):
        times = []
        hour = 6
        minute = 0
        while hour < 21:
            start = f"{hour:02d}:{minute:02d}"
            minute += 30
            if minute == 60:
                minute = 0
                hour += 1
            end = f"{hour:02d}:{minute:02d}"
            times.append((start, end))
        return times

    def categorize_boxes(self):
        self.time_boxes = []
        self.day_boxes = []
        self.event_boxes = []

        time_col_width = int(self.img_w * 0.18)  
        header_row_height = int(self.img_h * 0.10)

        # --- STEP A: Add Colored Event Blocks ---
        clean_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        for family_mask in self.color_families:
            opened = cv2.morphologyEx(family_mask, cv2.MORPH_OPEN, clean_kernel)
            contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if w > 20 and h > 20:  
                    pad = 4
                    self.event_boxes.append((x - pad, y - pad, w + pad*2, h + pad*2))

        # --- STEP B: Find Time Slots (Left Zone) ---
        time_zone = self.ocr_mask.copy()
        time_zone[:, time_col_width:] = 0 
        time_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 2))
        time_dilated = cv2.dilate(time_zone, time_kernel, iterations=1)
        time_contours, _ = cv2.findContours(time_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in time_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 10 and h > 5 and y > header_row_height: 
                self.time_boxes.append((x, y, w, h))

        # --- STEP C: Find Day Headers (Top Zone) ---
        day_zone = self.ocr_mask.copy()
        day_zone[header_row_height:, :] = 0 
        day_zone[:, :time_col_width] = 0 
        day_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 5))
        day_dilated = cv2.dilate(day_zone, day_kernel, iterations=1)
        day_contours, _ = cv2.findContours(day_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in day_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 40 and h > 10:
                self.day_boxes.append((x, y, w, h))

        # --- STEP D: Find Floating Text Events (Middle Zone - "Breaks", etc.) ---
        event_zone = self.ocr_mask.copy()
        event_zone[:header_row_height, :] = 0 
        event_zone[:, :time_col_width] = 0 
        
        event_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
        event_dilated = cv2.dilate(event_zone, event_kernel, iterations=1)
        event_contours, _ = cv2.findContours(event_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in event_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w < 10 or h < 10: 
                continue

            is_inside_color_box = False
            center_x, center_y = x + (w // 2), y + (h // 2)
            for (gx, gy, gw, gh) in self.event_boxes:
                if gx <= center_x <= gx + gw and gy <= center_y <= gy + gh:
                    is_inside_color_box = True
                    break
            
            if not is_inside_color_box:
                self.event_boxes.append((x, y, w, h))

    def ocr_crop(self, box, index):
        x, y, w, h = box
        pad = 4
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(self.img_w, x + w + pad), min(self.img_h, y + h + pad)

        crop = self.ocr_mask[y1:y2, x1:x2]
        crop_for_ocr = cv2.bitwise_not(crop)

        crop_h, crop_w = crop_for_ocr.shape[:2]
        min_dim = min(crop_w, crop_h)
        scale = 4.0 if min_dim < 30 else 3.0
        crop_for_ocr = cv2.resize(crop_for_ocr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        psm = 7 if (crop_h < 20 or crop_w < 60) else 6
        path = f"./process_images/ocr_table_tool/crop_{index}.jpg"
        cv2.imwrite(path, crop_for_ocr)

        output = subprocess.getoutput(f'tesseract "{path}" stdout --psm {psm}')
        return " ".join(output.strip().split())

    def clean_text(self, text):
        cleaned = re.sub(r'[^A-Za-z0-9\-\s]', ' ', text)
        compact_text = " ".join(cleaned.split())

        if not compact_text:
            return ""

        known_phrases = [
            "Administrative Hours", 
            "Consultation Hours", 
            "Research Hours", 
            "Community Extension Hours", 
            "Break"
        ]
        
        matches = difflib.get_close_matches(compact_text, known_phrases, n=1, cutoff=0.65)
        if matches:
            return matches[0]

        import re as _re
        words = compact_text.split()
        final_words = []
        known_prefixes = ['PSM', 'PSE', 'RM', 'PSY']

        for word in words:
            matched_prefix = None
            for pfx in known_prefixes:
                if word.upper().startswith(pfx) or (len(word) >= 3 and difflib.SequenceMatcher(None, word[:len(pfx)].upper(), pfx).ratio() > 0.7):
                    matched_prefix = pfx
                    break

            if matched_prefix and len(word) >= 4:
                suffix = word[len(matched_prefix):]
                suffix = suffix.replace('O', '0').replace('o', '0')
                suffix = suffix.replace('I', '1').replace('i', '1').replace('l', '1')
                suffix = suffix.replace('G', '6').replace('S', '5')
                word = matched_prefix + suffix

            if "-" in word:
                parts = word.split('-')
                if len(parts) == 2:
                    if parts[0] and parts[0][0] in ['i', 'I', 'l', 'L']:
                        parts[0] = '1' + parts[0][1:]
                    if parts[1] in ['8', 'b']:
                        parts[1] = 'B'
                    if parts[1] in ['0', 'o']:
                        parts[1] = 'D'
                    word = "-".join(parts)

            if len(word) <= 3 and word.islower() and word not in ['a', 'to', 'am', 'pm']:
                continue

            if _re.fullmatch(r'[^A-Za-z0-9]+', word):
                continue

            final_words.append(word)

        final_string = " ".join(final_words)

        matches2 = difflib.get_close_matches(final_string, known_phrases, n=1, cutoff=0.65)
        if matches2:
            return matches2[0]

        if len(final_words) > 5:
            short_words = sum(1 for w in final_words if len(w) <= 2)
            has_long_word = any(len(w) >= 5 for w in final_words)
            if short_words / len(final_words) > 0.6 and not has_long_word:
                return ""

        return final_string

    def extract_data(self):
            print("[System] Reading Day Headers and Building Grid...")

            time_slots = self.generate_time_slots()

            self.days = []
            for i, box in enumerate(self.day_boxes):
                text = self.ocr_crop(box, f"day_{i}")
                if len(text) >= 3:
                    self.days.append({
                        'text': text,
                        'center_x': box[0] + (box[2] // 2),
                        'bottom_y': box[1] + box[3]
                    })

            if not self.days:
                print("[Error] Could not find day headers to anchor the schedule.")
                return

            grid_top_y = sum(d['bottom_y'] for d in self.days) / len(self.days)
            grid_bottom_y = self.img_h
            total_grid_height = grid_bottom_y - grid_top_y
            row_height = total_grid_height / len(time_slots)

            print("\n" + "="*50)
            print(" EXTRACTED SCHEDULE EVENTS")
            print("="*50)

            for i, event in enumerate(self.event_boxes):
                x, y, w, h = event

                raw_event_text = self.ocr_crop(event, f"event_{i}")
                event_text = self.clean_text(raw_event_text)

                if len(event_text) < 4:
                    continue

                event_center_x = x + (w // 2)
                day_text = "Unknown Day"
                if self.days:
                    closest_day = min(
                        self.days,
                        key=lambda d: abs(d['center_x'] - event_center_x)
                    )
                    day_text = closest_day['text']

                start_index = int(round((y - grid_top_y) / row_height))
                end_index = int(round((y + h - grid_top_y) / row_height))

                start_index = max(0, min(start_index, len(time_slots) - 1))
                end_index = max(start_index + 1, min(end_index, len(time_slots)))

                start_time_text = time_slots[start_index][0]
                end_time_text = time_slots[end_index - 1][1]

                print(f"DAY:   {day_text}")
                print(f"TIME:  {start_time_text} to {end_time_text}")
                print(f"EVENT: {event_text}")
                print("-" * 50)

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    input_filename = "new_sched_2.jpg"
    
    color_output_filename = "schedule_final_colored.jpg"
    bw_output_filename = "schedule_final_bw_mask.jpg"

    image = cv2.imread(input_filename)

    if image is None:
        print(f"Error: Could not load '{input_filename}'.")
    else:
        os.makedirs("./process_images/table_lines_remover/", exist_ok=True)
        os.makedirs("./process_images/ocr_table_tool/", exist_ok=True)

        remover = TableLinesRemover(image)
        cleaned_and_cropped_image = remover.execute()
        
        cv2.imwrite(color_output_filename, cleaned_and_cropped_image)
        print(f"Success! Cleaned color image saved as '{color_output_filename}'")

        extractor = ScheduleDataExtractor(cleaned_and_cropped_image)
        
        cv2.imwrite(bw_output_filename, extractor.bw_image_for_viewing)
        print(f"Success! Black & White text mask saved as '{bw_output_filename}'")

        extractor.execute()