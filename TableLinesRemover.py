import cv2
import numpy as np
import os
import subprocess
import re

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

    def isolate_color_blocks(self):
        hsv = cv2.cvtColor(self.original, cv2.COLOR_BGR2HSV)
        
        # FIX 1: Lowered brightness to 30 to catch Dark Green, Dark Red, Blue, etc.
        lower_color = np.array([0, 30, 30])
        upper_color = np.array([180, 255, 255])
        raw_color_mask = cv2.inRange(hsv, lower_color, upper_color)
        
        contours, _ = cv2.findContours(raw_color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.solid_color_mask = np.zeros_like(raw_color_mask)
        for cnt in contours:
            if cv2.contourArea(cnt) > 200:
                cv2.drawContours(self.solid_color_mask, [cnt], -1, 255, -1)
        
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

        # 1. FIND THE COLORED BLOCKS (Red, Green, Yellow, Blue, etc.)
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        lower_color = np.array([0, 30, 30])
        upper_color = np.array([180, 255, 255])
        self.color_mask = cv2.inRange(hsv, lower_color, upper_color)

        # 2. CREATE PURE TEXT MASK
        text_only_image = self.image.copy()
        text_only_image[self.color_mask == 255] = [255, 255, 255] # Erase all colors
        text_only_gray = cv2.cvtColor(text_only_image, cv2.COLOR_BGR2GRAY)
        _, self.ocr_mask = cv2.threshold(text_only_gray, 130, 255, cv2.THRESH_BINARY_INV)
        
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
        # THE FIX: Removed the "MORPH_CLOSE" superglue. We now strictly Erode (Scalpel)
        # to slice apart touching columns (Thurs/Fri) and touching rows (Tuesday).
        shrink_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
        shrunk_color = cv2.erode(self.color_mask, shrink_kernel, iterations=1)

        contours, _ = cv2.findContours(shrunk_color, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 15 and h > 15:  
                # We eroded by an 8x8 kernel (shaving roughly 4 pixels off all sides)
                # We add 4 pixels of padding back here so we don't chop the text edges.
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
            if w > 10 and h > 5:
                self.day_boxes.append((x, y, w, h))

        # --- STEP D: Find Floating Text Events (Middle Zone - "Breaks", etc.) ---
        event_zone = self.ocr_mask.copy()
        event_zone[:header_row_height, :] = 0 
        event_zone[:, :time_col_width] = 0 
        
        event_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 15))
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
        crop_for_ocr = cv2.resize(crop_for_ocr, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)

        path = f"./process_images/ocr_table_tool/crop_{index}.jpg"
        cv2.imwrite(path, crop_for_ocr)

        output = subprocess.getoutput(f'tesseract "{path}" stdout --psm 6')
        return " ".join(output.strip().split())

    def clean_text(self, text):
        """Dynamically scrubs OCR noise using smarter pattern recognition."""
        cleaned = re.sub(r'[^A-Za-z0-9\-\s]', ' ', text)
        words = cleaned.split()
        final_words = []

        for word in words:
            # FIX 2: Only apply Course Code logic if the word actually LOOKS like a course code
            if len(word) >= 5 and word[:3].isalpha() and word[3] in '0123456789OoIiLl':
                prefix = word[:3]
                suffix = word[3:]
                new_suffix = ""
                for char in suffix:
                    if char in 'Oo': new_suffix += '0'
                    elif char in 'IiLl': new_suffix += '1'
                    else: new_suffix += char
                word = prefix + new_suffix

            # DYNAMIC RULE 2: Fix Sections (e.g., iBSN-A -> 1BSN-A)
            if "-" in word:
                parts = word.split('-')
                if len(parts) == 2 and len(parts[0]) > 2:
                    first_char = parts[0][0]
                    if first_char in ['i', 'I', 'l', 'L']:
                        word = '1' + word[1:]

            # DYNAMIC RULE 3: Drop noise speckles
            if len(word) == 1 and word.islower() and word != 'a':
                continue

            final_words.append(word)
            
        return " ".join(final_words)

    def extract_data(self):
        print("[System] Reading Day Headers...")

        # -------------------------------
        # STEP 1: Generate clean time labels
        # -------------------------------
        time_slots = self.generate_time_slots()

        # -------------------------------
        # STEP 2: Sort detected time boxes (TOP → BOTTOM)
        # -------------------------------
        sorted_time_boxes = sorted(self.time_boxes, key=lambda b: b[1])

        # --- THE TIME SHIFT FIX ---
        # The script misses the first 2 faint rows (6:00 and 6:30), 
        # so the first box it actually finds is 7:00 AM. 
        # We offset the time list by 2 so it starts mapping at 07:00 AM.
        slot_offset = 2 

        # Build mapping: each detected row → actual time slot
        time_map = []
        for i, box in enumerate(sorted_time_boxes):
            # Make sure we don't go out of bounds of the time_slots list
            if (i + slot_offset) >= len(time_slots):
                break
                
            y = box[1]
            # Apply the offset here
            start, end = time_slots[i + slot_offset]
            
            time_map.append({
                "y": y,
                "start": start,
                "end": end
            })

        # -------------------------------
        # STEP 3: Read Day Headers (UNCHANGED)
        # -------------------------------
        self.days = []
        for i, box in enumerate(self.day_boxes):
            text = self.ocr_crop(box, f"day_{i}")
            if len(text) >= 3:
                self.days.append({
                    'text': text,
                    'center_x': box[0] + (box[2] // 2)
                })

        print("\n" + "="*50)
        print(" EXTRACTED SCHEDULE EVENTS")
        print("="*50)

        # -------------------------------
        # STEP 4: Process Events
        # -------------------------------
        for i, event in enumerate(self.event_boxes):
            x, y, w, h = event

            # OCR EVENT TEXT
            raw_event_text = self.ocr_crop(event, f"event_{i}")
            event_text = self.clean_text(raw_event_text)

            if len(event_text) < 4:
                continue

            # -------------------------------
            # STEP 5: FIND DAY (UNCHANGED)
            # -------------------------------
            event_center_x = x + (w // 2)

            day_text = "Unknown Day"
            if self.days:
                closest_day = min(
                    self.days,
                    key=lambda d: abs(d['center_x'] - event_center_x)
                )
                day_text = closest_day['text']

            # -------------------------------
            # STEP 6: TIME MAPPING (FINAL FIX)
            # -------------------------------
            if not time_map:
                continue

            row_h = abs(time_map[1]['y'] - time_map[0]['y'])

            closest_start = min(time_map, key=lambda t: abs(t['y'] - (y + row_h/2)))
            closest_end   = min(time_map, key=lambda t: abs(t['y'] - (y + h - row_h/2)))

            start_time_text = closest_start['start']
            end_time_text = closest_end['end']

            # -------------------------------
            # OUTPUT
            # -------------------------------
            print(f"DAY:   {day_text}")
            print(f"TIME:  {start_time_text} to {end_time_text}")
            print(f"EVENT: {event_text}")
            print("-" * 50)

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    input_filename = "schedule_image_1.jpg"
    
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