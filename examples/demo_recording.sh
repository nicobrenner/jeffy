#!/bin/bash
# Demo recording script for README / Show HN
# Run: bash examples/demo_recording.sh
# Record with: asciinema rec demo.cast -- bash examples/demo_recording.sh

set -e

echo "# Jeffy: pretrained classifiers you can run and retrain on CPU"
echo ""
sleep 1

echo "$ python -c \""
echo "from jeffy.engine import Engine"
echo "engine = Engine()"
echo "engine.load()"
echo ""
echo "# Classify with a pretrained head"
echo "r = engine.predict('banking77', 'I was charged twice for the same transaction')"
echo "print(r['label'], f\\\"({r['confidence']:.0%})\\\")"
echo ""
echo "r = engine.predict('sms_spam', 'WINNER! Call now to claim your 1000 dollar prize!')"
echo "print(r['label'], f\\\"({r['confidence']:.0%})\\\")"
echo "\""
echo ""

python -c "
from jeffy.engine import Engine
engine = Engine()
engine.load()
r = engine.predict('banking77', 'I was charged twice for the same transaction')
print(r['label'], f\"({r['confidence']:.0%})\")
r = engine.predict('sms_spam', 'WINNER! Call now to claim your 1000 dollar prize!')
print(r['label'], f\"({r['confidence']:.0%})\")
" 2>/dev/null

echo ""
sleep 1
echo "# Train a custom classifier from examples"
echo ""

python -c "
from jeffy.train import train_classifier

clf = train_classifier(
    texts=[
        'Q3 budget review meeting tomorrow at 2pm',
        'Please review the attached contract',
        'Mom birthday dinner Saturday at 7',
        'Soccer practice canceled due to rain',
        '50% off everything this weekend only!',
        'New arrivals just dropped',
        'Your package has been delivered',
        'Password changed successfully',
        'Team standup moved to 10am',
        'Grandma sent photos from the reunion',
        'Flash sale: free shipping today',
        'Your flight check-in is now open',
    ],
    labels=[
        'work', 'work', 'family', 'family',
        'promo', 'promo', 'notification', 'notification',
        'work', 'family', 'promo', 'notification',
    ],
    task_id='inbox',
    test_size=0,
    cv_folds=0,
    verbose=False,
)

# Predictions on messages NOT in training data
print()
print('# Classify new messages (not in training data)')
for msg in [
    'Can you send me the Q4 projections?',
    'Dad wants to know about the cookout',
    'Buy 2 get 1 free, today only!',
    'Your credit card statement is ready',
]:
    r = clf.predict(msg)
    print(f'  {r[\"label\"]:15s} ({r[\"confidence\"]:.0%})  \"{msg}\"')
" 2>/dev/null

echo ""
echo "# All on CPU. No API keys. ~100ms per prediction."
