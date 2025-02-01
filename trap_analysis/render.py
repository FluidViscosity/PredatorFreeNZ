import os
from jinja2 import Template

# from weasyprint import HTML

# HTML template as a string
html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
        }
        .container {
            display: flex;
            justify-content: space-between;
            align-items: center;
            height: 100vh;
            padding: 20px;
            box-sizing: border-box;
        }
        .text-content {
            width: 50%;
        }
        .map-content {
            width: 45%;
            text-align: center;
            border: 1px solid #ccc;
            overflow: hidden;
        }
        .title {
            font-size: 36px;
            font-weight: bold;
            margin-bottom: 20px;
        }
        .description {
            font-size: 16px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="text-content">
            <div class="title">{{ title }}</div>
            <div class="description">{{ description }}</div>
        </div>
        <div class="map-content">
            {{ map_content | safe }}
        </div>
    </div>
</body>
</html>
"""


def render_html(title, description, map_content, output_html):
    template = Template(html_template)
    html_content = template.render(
        title=title, description=description, map_content=map_content
    )

    with open(output_html, "w") as f:
        f.write(html_content)


def read_map_file(map_file):
    with open(map_file, "r") as f:
        return f.read()


# def html_to_image(html_file, output_image):
#     HTML(html_file).write_png(output_image)


# Function to create a PowerPoint presentation and add images
# def create_presentation(image_files, pptx_file):
#     prs = Presentation()

#     for image in image_files:
#         slide = prs.slides.add_slide(prs.slide_layouts[5])  # Layout 5 is a blank slide
#         left = Inches(1)
#         top = Inches(1)
#         slide.shapes.add_picture(image, left, top, width=Inches(8.5))

#     prs.save(pptx_file)


if __name__ == "__main__":
    # Example data
    pages_data = [
        {
            "title": "Title 1",
            "description": "This is a description for the first image.",
            "map_file": "Buffer - Boundary.html",
            "caption": "Caption for Image 1",
        },
        {
            "title": "Title 2",
            "description": "This is a description for the second image.",
            "map_file": "Buffer - Boundary.html",
            "caption": "Caption for Image 2",
        },
    ]
    # Directory setup
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # Generate HTML files and convert to images
    image_files = []
    for i, page_data in enumerate(pages_data):
        html_file = os.path.join(output_dir, f"page_{i+1}.html")
        image_file = html_file.replace(".html", ".png")

        # Read the map file content
        map_content = read_map_file(page_data["map_file"])

        # Render HTML
        render_html(
            title=page_data["title"],
            description=page_data["description"],
            map_content=map_content,
            output_html=html_file,
        )

        # Convert HTML to image
        # html_to_image(html_file, image_file)
        # image_files.append(image_file)
