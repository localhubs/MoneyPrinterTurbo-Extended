<div align="center">
<h1 align="center">MoneyPrinterTurbo - Enhanced Fork</h1>

<p align="center">
  <a href="https://github.com/harry0703/MoneyPrinterTurbo/stargazers"><img src="https://img.shields.io/github/stars/harry0703/MoneyPrinterTurbo.svg?style=for-the-badge" alt="Stargazers"></a>
  <a href="https://github.com/harry0703/MoneyPrinterTurbo/issues"><img src="https://img.shields.io/github/issues/harry0703/MoneyPrinterTurbo.svg?style=for-the-badge" alt="Issues"></a>
  <a href="https://github.com/harry0703/MoneyPrinterTurbo/network/members"><img src="https://img.shields.io/github/forks/harry0703/MoneyPrinterTurbo.svg?style=for-the-badge" alt="Forks"></a>
  <a href="https://github.com/harry0703/MoneyPrinterTurbo/blob/main/LICENSE"><img src="https://img.shields.io/github/license/harry0703/MoneyPrinterTurbo.svg?style=for-the-badge" alt="License"></a>
</p>
</div>

## 🎬 Example Videos

See MoneyPrinterTurbo in action! These videos showcase the enhanced subtitle highlighting and semantic video selection features:

<div align="center">

### 📺 Full-Length Video Example
[![MoneyPrinterTurbo Example Video](https://img.youtube.com/vi/yXc07ROgj80/maxresdefault.jpg)](https://www.youtube.com/watch?v=yXc07ROgj80)

**[🎥 Watch Full Video →](https://www.youtube.com/watch?v=yXc07ROgj80)**

---

### 📱 YouTube Shorts Example  
[![MoneyPrinterTurbo Shorts Example](https://img.youtube.com/vi/JBAuXpVHt40/maxresdefault.jpg)](https://www.youtube.com/shorts/JBAuXpVHt40)

**[🎥 Watch Short →](https://www.youtube.com/shorts/JBAuXpVHt40)**

</div>

*These videos demonstrate the enhanced word-by-word subtitle highlighting, center-aligned text, intelligent video-text matching, and professional-quality output that MoneyPrinterTurbo can generate automatically.*

---

## Original Repo Credit

This is an **enhanced fork** of the amazing [MoneyPrinterTurbo](https://github.com/harry0703/MoneyPrinterTurbo) project. Check out the original repo and **full credit goes to the original author and contributors**. This fork adds advanced subtitle highlighting features while maintaining all the original functionality.

---

## New Features in This Fork

### Word-by-Word Subtitle Highlighting

This enhanced version introduces **intelligent word-level highlighting** in subtitles, making videos more engaging and easier to follow:

- **Real-time Word Highlighting**: Each word is highlihted exactly when it's being spoken
- **Normal Text Color**: Non-spoken words remain in the original subtitle color 
- **Microsoft TTS2 Integration**: Perfect synchronization with Microsoft's Text-to-Speech timing
- **Customizable Colors**: Configure highlight colors through the web interface
- **Multi-line Support**: Works seamlessly with wrapped text and multiple subtitle lines

### Enhanced Video-Text Alignment 

- **Text-based Similarity**: Current semantic search analyzes script content to match relevant video clips
- **Video-Thumbnail Similarity**: Current semantic search we can enable video thubnail(of video content) similarity for video sources like 'Pexels'. This when enabled is combined with text simialrity with 30% text + 70% thumbnail cimialrity with text produce the best results so far  
- **Better than Sequential**: No manual effort required (unlike sequential mode)
- **Better than Random**: Much more relevant than random video selection

**Future Roadmap:**
- **Video Content Analysis**: AI-powered analysis of actual video content for semantic matching
- **WhisperX Integration**: Enhanced subtitle timing with WhisperX for even more precise word highlighting, this will also help us get rid of Microsoft TTS for precise word boundaries.

## Installation & Setup

### Prerequisites

- Python 3.11 or higher
- Git
- Conda (recommended) or Python virtual environment

### Quick Start

1. **Clone the repository:**
```bash
git clone https://github.com/[YOUR_USERNAME]/MoneyPrinterTurbo.git
cd MoneyPrinterTurbo
```

2. **Create and activate conda environment:**
```bash
conda create -n MoneyPrinterTurbo python=3.11
conda activate MoneyPrinterTurbo
```

3. **Install dependencies:**
```bash
# Install all dependencies (includes optimization libraries for caching & performance)
pip install -r requirements.txt
```

####   Alternative: One-liner Environment Setup

```bash
conda env create -f MoneyPrinterTurbo_environment.yml 
conda activate MoneyPrinterTurbo
```

4. **Run the web interface:**
```bash
# On Linux/MacOS
sh webui.sh

# On Windows
webui.bat
```

The web interface will automatically open in your browser at `http://localhost:7860`


---
