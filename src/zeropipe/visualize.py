from zeropipe.schedules.schedule import FnType, ScheduleNode

HTML_HEADER = """<!DOCTYPE html>
<html>

<head>
    <style>
        .timeline-container {
            position: relative;
            padding: 20px 0;
            max-width: 90%;
            margin: 0 auto;
        }

        .timeline-wrapper {
            display: flex;
            align-items: stretch;
        }

        .stream-labels {
            width: 100px;
            flex-shrink: 0;
            margin-right: 20px;
            padding-top: 40px;
            /* Space for time markers */
        }

        .stream-label {
            height: 40px;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            font-weight: bold;
            color: #333;
        }

        .timeline-content {
            flex-grow: 1;
        }

        .timeline-stream {
            position: relative;
            height: 40px;
            margin-bottom: 20px;
            background-color: #f0f0f0;
            border-radius: 4px;
        }

        .timeline-task {
            position: absolute;
            height: 30px;
            top: 5px;
            border-radius: 4px;
            padding: 5px;
            box-sizing: border-box;
            color: white;
            font-size: 12px;
            transition: all 0.3s ease;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .timeline-task:hover {
            transform: scale(1.05);
            z-index: 1;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
        }

        .time-markers {
            position: relative;
            height: 20px;
            border-bottom: 1px solid #ccc;
            margin-bottom: 20px;
        }

        .time-marker {
            position: absolute;
            transform: translateX(-50%);
            font-size: 12px;
        }

        .time-marker::before {
            content: '';
            position: absolute;
            height: 5px;
            width: 1px;
            background-color: #ccc;
            bottom: -5px;
            left: 50%;
        }
        
        .forward { background-color: #2E7D32; }   /* Darker Green */
        .backward { background-color: #FFA000; }  /* Darker Yellow */
        .weight { background-color: #1976D2; }    /* Darker Blue */
        .send { background-color: #D32F2F; }     /* Darker Red */
        .overlapped { background-color: #F57C00; }  /* Darker Orange */
        .recv { background-color: #7B1FA2; }  /* Darker Purple */

        .forward-soft { background-color: #4CAF50; }  /* Material Green */
        .backward-soft { background-color: #FFC107; } /* Material Amber */
        .weight-soft { background-color: #2196F3; }   /* Material Blue */
        .send-soft { background-color: #F44336; }    /* Material Red */
        .overlapped-soft { background-color: #FF9800; } /* Material Orange */
        .recv-soft { background-color: #9C27B0; } /* Material Purple */
        
        /* Legend styles */
        .timeline-legend {
            display: flex;
            gap: 16px;
            padding: 8px;
            background: #f5f5f5;
            border-radius: 4px;
            margin-top: 12px;
            flex-wrap: wrap;
        }

        .legend-item {
            display: flex;
            align-items: center;
            gap: 6px;
            color: #333;
            padding: 4px 8px;
            border-radius: 4px;
            transition: background-color 0.2s ease;
        }

        .legend-item:hover {
            background-color: #e0e0e0;
            cursor: pointer;
        }

        .legend-color {
            width: 12px;
            height: 12px;
            border-radius: 2px;
            flex-shrink: 0;
            transition: transform 0.2s ease;
        }

        .legend-item:hover .legend-color {
            transform: scale(1.1);
        }
    </style>
</head>

<body>

    <div class="timeline-container">
        <div class="timeline-wrapper">
            <!-- Stream Labels -->"""

HTML_FOOTER = """
        </div>
        
        <!-- Legend -->
        <div class="timeline-legend">
            <div class="legend-item" data-type="Forward">
                <div class="legend-color forward"></div>
                <span>Forward</span>
            </div>
            <div class="legend-item" data-type="Backward">
                <div class="legend-color backward"></div>
                <span>Backward</span>
            </div>
            <div class="legend-item" data-type="Weight">
                <div class="legend-color weight"></div>
                <span>Weight</span>
            </div>
            <div class="legend-item" data-type="Send">
                <div class="legend-color send"></div>
                <span>Send</span>
            </div>
            <div class="legend-item" data-type="Recv">
                <div class="legend-color recv"></div>
                <span>Recv</span>
            </div>
        </div>
    </div>

</body>

</html>"""


def visualize_schedule(output_path: str, schedule: list[list[ScheduleNode]]):
    fn_type_to_color = {
        FnType.F: "forward",
        FnType.B: "backward",
        FnType.W: "weight",
        FnType.SEND_FORWARD: "send",
        FnType.RECV_FORWARD: "recv",
        FnType.SEND_BACKWARD: "send",
        FnType.RECV_BACKWARD: "recv",
    }

    stream_labels = []
    for stage in range(len(schedule)):
        stream_labels.append(f"<div class='stream-label'>Compute {stage}</div>")
        stream_labels.append(f"<div class='stream-label'>Communicate {stage}</div>")

    stream_labels_str = (
        "<div class='stream-labels'>\n" + "\n".join(stream_labels) + "</div>"
    )

    max_time = max(node.complete_time for stage in schedule for node in stage)
    timestreams = []

    node_type_title = {
        FnType.F: "Forward",
        FnType.B: "Backward",
        FnType.W: "Weight",
        FnType.SEND_FORWARD: "Send Forward",
        FnType.RECV_FORWARD: "Recv Forward",
        FnType.SEND_BACKWARD: "Send Backward",
        FnType.RECV_BACKWARD: "Recv Backward",
    }

    for stage in schedule:
        timeline_compute = []
        timeline_communicate = []

        for node in stage:
            color = fn_type_to_color[node.type]

            if node.chunk == 1:
                color = f"{color}-soft"

            left = round(node.start_time / max_time * 100, 3)
            width = round((node.complete_time - node.start_time) / max_time * 100, 3)

            title = f"{node_type_title[node.type]} Start: {node.start_time}ms, End: {node.complete_time}ms"

            if node.type.is_compute():
                timeline_compute.append(
                    f"<div class='timeline-task {color}' style='left: {left}%; width: {width}%;' title='{title}'>{node.microbatch}</div>"
                )
            else:
                timeline_communicate.append(
                    f"<div class='timeline-task {color}' style='left: {left}%; width: {width}%;' title='{title}'>{node.microbatch}</div>"
                )

        compute_str = "\n".join(timeline_compute)
        communicate_str = "\n".join(timeline_communicate)

        timestreams.append(f"<div class='timeline-stream'>\n{compute_str}\n</div>")
        timestreams.append(f"<div class='timeline-stream'>\n{communicate_str}\n</div>")

    timemarkers_str = f"""<div class="time-markers">
        <div class="time-marker" style="left: 0%">0ms</div>
        <div class="time-marker" style="left: 25%">{round(max_time * 0.25)}ms</div>
        <div class="time-marker" style="left: 50%">{round(max_time * 0.5)}ms</div>
        <div class="time-marker" style="left: 75%">{round(max_time * 0.75)}ms</div>
        <div class="time-marker" style="left: 100%">{max_time}ms</div>
    </div>"""

    timestreams_str = "\n".join(timestreams)

    timelines_str = (
        "<div class='timeline-content'>\n"
        + timemarkers_str
        + timestreams_str
        + "</div>"
    )

    with open(output_path, "w") as f:
        f.write(HTML_HEADER + stream_labels_str + timelines_str + HTML_FOOTER)
