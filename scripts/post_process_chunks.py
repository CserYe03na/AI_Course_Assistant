import json
import re
from pathlib import Path
import argparse

# 1. Clean text
def clean_text(text):
    if not text:
        return ""

    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    #text = re.sub(r'[^\w\s]', '', text)
    # remove noisy patterns
    text = text.encode("ascii", "ignore").decode()
    text = re.sub(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', '', text)
    text = re.sub(r'(?:\b[a-z]\s+){2,}[a-z]\b', lambda m: m.group(0).replace(" ", ""), text)
    

    return text.strip()



def is_useless(text):
    bad_keywords = [
        "office hours", "email", "courseworks", "grading",
        "midterm", "final", "assignment", "policy", "schedule",
        "columbia university", "professor", "andrei",
        "use ed", "zoom",
        "todo", "quiz", "due"
    ]

    if any(k in text for k in bad_keywords):
        return True

    if re.search(r'in\s*\[\d+\]|out\s*\[\d+\]', text):
        return True

    # remove broken latex / formulas
    if "\\frac" in text and text.count("{") < 2:
        return True

    # remove plot axis junk
    if re.search(r'\b\d+\.\d+\s+\d+\.\d+', text):
        return True
    
    # remove table-like content
    if "|" in text and text.count("|") > 4:
        return True

    # remove highly repetitive structured rows
    if re.search(r'(\w+\s+\|){3,}', text):
        return True

    return False

def assign_topic(text):
    text = text.lower()

    # ML tasks
    if "supervised" in text and "unsupervised" in text:
        return "ml_tasks"
    
    if "regression" in text and "classification" in text:
        return "ml_tasks"

    # specific models
    if "knn" in text:
        return "knn"
    
    if "decision tree" in text:
        return "decision_tree"
    
    if any(k in text for k in ["ensemble", "bagging", "boosting"]):
        return "ensemble"

    # single concepts
    if "regression" in text:
        return "regression"
    
    if "classification" in text:
        return "classification"

    # tools
    if any(k in text for k in ["numpy", "pandas", "python"]):
        return "tools"

    # general ML
    if any(k in text for k in ["model", "learning", "prediction"]):
        return "ml_concepts"
    
    #formulas
    if any(k in text for k in ["\\frac", "logistic", "exp", "e ^"]):
        return "formula"

    return "other"


# 2. Merge blocks → chunks
def create_chunks(data, min_len=40, max_len=400):
    chunks = []

    for doc in data["documents"]:
        doc_id = doc["doc_id"]
        lecture = doc["source_file"].split("/")[-1]

        for page in doc["pages"]:
            page_no = page["page_no"]
            has_image = any(b.get("image_path") for b in page["blocks"])
            image_blocks = [b for b in page["blocks"] if b.get("image_path")]
            is_image_page = len(image_blocks) > 0 and has_image
            chunk_idx = 0
            # handle image-heavy page (IMPORTANT)
            if is_image_page:
                chunk_idx += 1

                # extract title
                title_text = ""

                for b in page["blocks"]:
                    if b.get("type") == "title":
                        title_text = clean_text(b.get("text"))
                        break

                if not title_text:
                    for b in page["blocks"]:
                        t = clean_text(b.get("text"))
                        if len(t.split()) > 5:
                            title_text = t[:80]
                            break

                #  summary
                if title_text:
                    summary = f"{title_text} diagram"
                else:
                    summary = f"{lecture} page {page_no} contains a machine learning diagram"

                chunks.append({
                    "chunk_id": f"{lecture}_p{page_no}_c{chunk_idx}",
                    "content": summary,
                    "page": page_no,
                    "lecture": lecture,
                    "topic": "diagram",
                    "has_image": True
                })

                continue
            # collect all clean text in page
            texts = []

            for block in page["blocks"]:
                text = clean_text(block.get("text"))
              

                if not text:
                    continue
                
                if is_useless(text):
                    continue

                if re.search(r'(\d+\s+){5,}', text):
                    continue

                texts.append(text)

            if not texts:
                continue

            # merge whole page
            full_text = " ".join(texts)

            # split into sentences
            sentences = re.split(r'\.\s+|\n|•| - ', full_text)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 0]

            buffer = ""

            for sent in sentences:
                if len(sent) < 20:
                    continue

                if len(buffer.split()) + len(sent.split()) < max_len:
                    buffer += " " + sent
                else:
                    if len(buffer.split()) > min_len or len(buffer.split()) > 20:
                        chunk_idx += 1
                        chunks.append({
                            "chunk_id": f"{lecture}_p{page_no}_c{chunk_idx}",
                            "content": buffer.strip(),
                            "page": page_no,
                            "lecture": lecture,
                            "topic": assign_topic(buffer),
                            "has_image": has_image
                        })
                    buffer = sent

            #last chunk
            if len(buffer.split()) > min_len or len(buffer.split()) > 20:
                chunk_idx += 1
                chunks.append({
                    "chunk_id": f"{lecture}_p{page_no}_c{chunk_idx}",
                     "content": buffer.strip(),
                     "page": page_no,
                     "lecture": lecture,
                     "topic": assign_topic(buffer),
                     "has_image": has_image
                        })

    return chunks

def deduplicate(chunks):
    seen = set()
    unique = []

    for c in chunks:
        key = " ".join(c["content"].split()[:40])
        if key not in seen:
            unique.append(c)
            seen.add(key)

    return unique

# 4. Run pipeline
def main():
    
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--course-id",
        default="eods",
        help="Course identifier"
    )

    parser.add_argument(
        "--course-name",
        default="Elements of Data Science",
        help="Course name"
    )

    args = parser.parse_args()

    course_id = args.course_id
    course_name = args.course_name  

    input_path = Path(f"data/processed/{course_id}/{course_id}_document.json")

    output_dir = Path(f"data/processed/{course_id}")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{course_id}_chunks.json"

    with open(input_path) as f:
        data = json.load(f)
    
    chunks = create_chunks(data)
    chunks = deduplicate(chunks)

    print("Total chunks:", len(chunks))

    output = {
        "course_id": course_id,
        "course_name": course_name,
        "documents": chunks
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()