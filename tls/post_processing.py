"""
The MIT License (MIT)

Copyright (c) 2017-2018 Sebastian Martschat

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import datetime


def post_process(ranked_sentences,
                 extents,
                 daily_summary_length,
                 timeline_length,
                 start,
                 end,
                 knee=False):
    dates_to_chosen_sentences = {}
    forbidden_dates = set()
    for i, sent in enumerate(ranked_sentences):
        if len(sent.time) > 0:
            date = sent.get_date().date()
        else:
            date = sent.pub_time.date()

        if date < start.date() or date > end.date():
            continue

        if date in dates_to_chosen_sentences and len(dates_to_chosen_sentences[date]) == daily_summary_length:
            continue

        if date in forbidden_dates:
            continue

        if not knee and len(dates_to_chosen_sentences) == timeline_length:
            if date in dates_to_chosen_sentences and len(dates_to_chosen_sentences[date]) < daily_summary_length:
                pass
            else:
                continue

        if extents is not None:
            forbidden_dates.add(date)
            try:
                for diff in range(1, extents[i] + 1):
                    forbidden_dates.add(date + datetime.timedelta(days=diff))
                    forbidden_dates.add(date + datetime.timedelta(days=-diff))
            except OverflowError:
                pass

        if date not in dates_to_chosen_sentences:
            dates_to_chosen_sentences[date] = []

        dates_to_chosen_sentences[date].append(sent)

    return dates_to_chosen_sentences
