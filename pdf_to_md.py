import base64
import os
from mistralai import Mistral
import html
from typing import Optional, Tuple


class PDFToMarkdown:
    def __init__(self, api_key: Optional[str] = None, client: Optional[Mistral] = None):
        if client is not None:
            self.client = client
        else:
            key = api_key or os.getenv('MISTRAL_API_KEY')
            self.client = Mistral(api_key=key)

    @staticmethod
    def encode_file(file_path: str) -> str:
        with open(file_path, "rb") as pdf_file:
            return base64.b64encode(pdf_file.read()).decode('utf-8')

    @staticmethod
    def fix_html_entities(text: str) -> str:
        return html.unescape(text)

    @staticmethod
    def save_base64_image(base64_data: str, output_path: str) -> None:
        if ',' in base64_data:
            base64_data = base64_data.split(',', 1)[1]

        image_bytes = base64.b64decode(base64_data)
        with open(output_path, 'wb') as f:
            f.write(image_bytes)

    def process(self, pdf_path: str, output_dir: str) -> Tuple[str, str]:
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        output_folder = os.path.join(output_dir, pdf_name)
        images_folder = os.path.join(output_folder, "images")

        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(images_folder, exist_ok=True)

        base64_file = self.encode_file(pdf_path)

        ocr_response = self.client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "document_url",
                "document_url": f"data:application/pdf;base64,{base64_file}"
            },
            include_image_base64=True,
            extract_header=True,
            extract_footer=True,
        )

        full_markdown = []
        image_counter = 0

        for page in ocr_response.pages:
            page_markdown = page.markdown

            for image in page.images:
                image_id = image.id
                image_base64 = image.image_base64

                if image_id.endswith('.jpeg') or image_id.endswith('.jpg'):
                    ext = '.jpeg'
                elif image_id.endswith('.png'):
                    ext = '.png'
                else:
                    if 'image/png' in image_base64:
                        ext = '.png'
                    else:
                        ext = '.jpeg'

                image_filename = f"image_{page.index}_{image_counter}{ext}"
                image_path = os.path.join(images_folder, image_filename)

                self.save_base64_image(image_base64, image_path)
                print(f"Saved image: {image_path}")

                relative_image_path = f"images/{image_filename}"
                page_markdown = page_markdown.replace(f"({image_id})", f"({relative_image_path})")

                image_counter += 1

            # if page.header:
            #     full_markdown.append(f"<!-- Header: {page.header.strip()} -->\n")

            full_markdown.append(page_markdown)

            # if page.footer:
            #     full_markdown.append(f"\n<!-- Footer: {page.footer.strip()} -->")

            # Add a newline between pages to prevent words merging, but avoid breaking flow too much
            full_markdown.append("\n") 
            # full_markdown.append("\n\n---\n\n")

        final_markdown = "".join(full_markdown)

        final_markdown = self.fix_html_entities(final_markdown)

        md_path = os.path.join(output_folder, f"{pdf_name}.md")
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(final_markdown)

        print(f"\nMarkdown saved to: {md_path}")
        print(f"Images saved to: {images_folder}")
        print(f"Total images saved: {image_counter}")

        return md_path, images_folder


if __name__ == "__main__":
    file_path = "in/leph105.pdf"
    output_dir = "output"

    converter = PDFToMarkdown()
    md_path, images_path = converter.process(file_path, output_dir)
    print(f"\nProcessing complete!")
