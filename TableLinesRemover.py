import cv2
import numpy as np
import os
import subprocess
import re
import difflib # <--- ADD THIS LINE

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
        """Dynamically scrubs OCR noise using fuzzy matching and smarter regex."""
        # Clean weird symbols but keep letters, numbers, hyphens, and spaces
        cleaned = re.sub(r'[^A-Za-z0-9\-\s]', ' ', text)
        compact_text = " ".join(cleaned.split())

        if not compact_text:
            return ""

        # --- DYNAMIC RULE 1: Fuzzy Match Known Schedule Phrases ---
        # This instantly fixes "Adm1n1strat1ve", "Sreak", and minor OCR typos.
        known_phrases = [
            "Administrative Hours", 
            "Consultation Hours", 
            "Research Hours", 
            "Community Extension Hours", 
            "Break"
        ]
        
        # Check if the text is at least 55% similar to a known phrase
        matches = difflib.get_close_matches(compact_text, known_phrases, n=1, cutoff=0.55)
        if matches:
            return matches[0]

        # --- DYNAMIC RULE 2: Process Course Codes and Sections ---
        words = compact_text.split()
        final_words = []

        for word in words:
            # Fix 1: Safely target Course Codes (e.g., PSM107, CHM104A)
            # Must be 5-8 chars and start with UPPERCASE letters to avoid normal words.
            if 5 <= len(word) <= 8 and word[:3].isupper():
                prefix = word[:3]
                suffix = word[3:]
                # Fix common OCR number/letter mix-ups only in the suffix
                new_suffix = suffix.replace('O', '0').replace('o', '0').replace('I', '1').replace('i', '1').replace('l', '1')
                word = prefix + new_suffix

            # Fix 2: Fix Sections (e.g., iPSY-A -> 1PSY-A, 2PSY-8 -> 2PSY-B)
            if "-" in word:
                parts = word.split('-')
                if len(parts) == 2:
                    if parts[0] and parts[0][0] in ['i', 'I', 'l', 'L']:
                        parts[0] = '1' + parts[0][1:]
                    if parts[1] == '8':
                        parts[1] = 'B'
                    word = "-".join(parts)

            # Fix 3: Drop noise speckles (1-2 character lowercase gibberish)
            if len(word) <= 2 and word.islower() and word not in ['a', 'to']:
                continue

            final_words.append(word)

        final_string = " ".join(final_words)

        # --- DYNAMIC RULE 3: Garbage Filter (The Saturday Issue) ---
        # The Saturday block is generating noise from empty table grid lines. 
        # Gibberish usually breaks down into lots of tiny 1-3 letter chunks.
        # If we have a long string of mostly tiny words, we erase it.
        if len(final_words) > 4 and all(len(w) < 4 for w in final_words):
            return "" 

        return final_string

    def extract_data(self):
            print("[System] Reading Day Headers and Building Grid...")

            # -------------------------------
            # STEP 1: Generate clean time labels (30 slots)
            # -------------------------------
            time_slots = self.generate_time_slots()

            # -------------------------------
            # STEP 2: Read Day Headers
            # -------------------------------
            self.days = []
            for i, box in enumerate(self.day_boxes):
                text = self.ocr_crop(box, f"day_{i}")
                if len(text) >= 3:
                    self.days.append({
                        'text': text,
                        'center_x': box[0] + (box[2] // 2),
                        'bottom_y': box[1] + box[3] # Grab the bottom edge of the day text
                    })

            if not self.days:
                print("[Error] Could not find day headers to anchor the schedule.")
                return

            # -------------------------------
            # STEP 3: DYNAMIC GEOMETRIC GRID (The Fix)
            # -------------------------------
            # Find the average bottom Y of the day headers. This is where 06:00 AM starts.
            grid_top_y = sum(d['bottom_y'] for d in self.days) / len(self.days)
            
            # The grid ends at the bottom of the cropped image
            grid_bottom_y = self.img_h
            
            # Divide the total pixel height by 30 slots to find the exact height of 30 minutes
            total_grid_height = grid_bottom_y - grid_top_y
            row_height = total_grid_height / len(time_slots)

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

                # FIND DAY
                event_center_x = x + (w // 2)
                day_text = "Unknown Day"
                if self.days:
                    closest_day = min(
                        self.days,
                        key=lambda d: abs(d['center_x'] - event_center_x)
                    )
                    day_text = closest_day['text']

                # -------------------------------
                # STEP 5: MATH-BASED TIME MAPPING
                # -------------------------------
                # We determine the start and end by dividing the Y pixels by the row height
                start_index = int(round((y - grid_top_y) / row_height))
                end_index = int(round((y + h - grid_top_y) / row_height))

                # Clamp indices to ensure they don't crash if an event touches the exact bottom edge
                start_index = max(0, min(start_index, len(time_slots) - 1))
                end_index = max(start_index + 1, min(end_index, len(time_slots)))

                start_time_text = time_slots[start_index][0]
                
                # The time_slots tuple holds (start_time, end_time). We grab the end_time of the final block.
                end_time_text = time_slots[end_index - 1][1]

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