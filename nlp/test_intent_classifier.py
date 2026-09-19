from nlp.intent_classifier import get_classifier
TEST_CASES = [
    # (متن, لیبل مورد انتظار)
    ("Docker container چطور با volume به هاست وصل میشه؟", "technical_question"),
    ("یه تابع پایتون بنویس که فایل csv رو بخونه", "code_request"),
    ("سلام چطوری امروز؟", "casual_chat"),
    ("I'm very angry and I want to draft a complaint letter.", "complaint_or_error"),
    ("API من 500 error میده وقتی رکوئست POST میفرستم", "complaint_or_error"),
    ("Terraform state file رو چطور مدیریت کنم بین چند نفر؟", "technical_question"),
    ("چیزی نگفتی که فهمم چیکار کنم، اصلا کار نمیکنه این کد", "complaint_or_error"),
]


def run():
    clf = get_classifier()
    correct = 0

    for text, expected in TEST_CASES:
        result = clf.classify(text)
        ok = result["intent"] == expected
        correct += ok

        status = "OK " if ok else "FAIL"
        print(f"[{status}] intent={result['intent']:<20} "
              f"conf={result['confidence']:.2f}  expected={expected}")
        print(f"      text: {text}")

    print(f"\n{correct}/{len(TEST_CASES)} correct")


if __name__ == "__main__":
    run()
