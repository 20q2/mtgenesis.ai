# MTGenesis.AI

🎴 An AI-powered Magic: The Gathering card generator that creates custom MTG cards with AI-generated text and artwork.

## 🌟 Features

- **AI-Generated Card Text**: Create unique card abilities using Ollama (local LLM)
- **Placeholder Artwork**: Simple placeholder images (AI image generation ready to add)
- **Complete MTG Card Creation**: Design cards with proper mana costs, types, rarities, and power/toughness
- **Real-time Preview**: See your card design update in real-time as you edit
- **Mana Cost Builder**: Visual mana symbol selector with automatic CMC calculation
- **Color Identity Detection**: Automatically detect card colors from mana cost
- **Card Type Support**: Full support for all MTG card types (Creature, Instant, Sorcery, etc.)
- **Rarity System**: Choose from Common, Uncommon, Rare, and Mythic Rare
- **Subtype Suggestions**: Context-aware subtype recommendations based on card type

## 🚀 Getting Started

### Prerequisites

- **Node.js** (v16 or higher)
- **npm**
- **Angular CLI** (v16+)
- **Python 3.11+**
- **Ollama** (for AI text generation)

### Installation & Setup

#### 1. Clone the Repository
```bash
git clone <repository-url>
cd mtgenesis.ai
```

#### 2. Install Frontend Dependencies
```bash
npm install
```

#### 3. Install Backend Dependencies
```bash
cd proxy-server
pip install -r requirements.txt
cd ..
```

#### 4. Install & Setup Ollama
1. **Download Ollama**: https://ollama.com/
2. **Install Ollama** on your system
3. **Pull the Mistral model**:
   ```bash
   ollama pull mistral:latest
   ```

### 🚀 Running the Application

You need to run **3 separate processes**:

#### Terminal 1: Start Ollama Server
```bash
ollama serve
```
*Keep this running - it provides AI text generation*

#### Terminal 2: Start Flask Backend
```bash
cd proxy-server
python app.py
```
*Runs on http://localhost:5000 - handles card generation API*

#### Terminal 3: Start Angular Frontend
```bash
ng serve
```
*Runs on http://localhost:4200 - the main web interface*

### 🎯 Usage

1. Open your browser to **http://localhost:4200**
2. Fill in the card form:
   - **Name**: Your card's name
   - **Type**: Creature, Instant, Sorcery, etc.
   - **Mana Cost**: Use the symbol buttons or type manually
   - **Colors**: Auto-detected from mana cost
   - **Rarity**: Common, Uncommon, Rare, Mythic
   - **Description**: Card abilities (optional - AI will generate)
   - **Art Prompt**: Description for artwork
3. Click **"Generate Card"**
4. View your generated card in the preview panel!

## 🛠️ Project Structure

```
mtgenesis.ai/
├── src/                           # Angular frontend
│   ├── app/
│   │   ├── components/
│   │   │   ├── card-form/         # Card creation form
│   │   │   └── card-preview/      # Card display component
│   │   ├── models/
│   │   │   ├── card.model.ts      # Card data structures
│   │   │   └── api.model.ts       # API interfaces
│   │   ├── services/
│   │   │   ├── card.service.ts    # API communication
│   │   │   └── mana.service.ts    # Mana cost utilities
│   │   └── app.component.ts       # Main application
│   └── environments/              # Configuration
└── proxy-server/                  # Flask backend
    ├── app.py                     # Main Flask application
    └── requirements.txt           # Python dependencies
```

## 🔧 Architecture

- **Frontend**: Angular 16 with Material UI
- **Backend**: Flask with CORS enabled
- **AI Text**: Ollama with Mistral model
- **AI Images**: Placeholder (ready for diffusers integration)
- **Communication**: Single unified API endpoint

## 🧪 API Endpoint

**POST** `http://localhost:5000/api/v1/create_card`

**Request Body:**
```json
{
  "prompt": "Fantasy art of dragon creature",
  "width": 384,
  "height": 288,
  "cardData": {
    "name": "Ancient Dragon",
    "colors": ["R"],
    "type": "Creature",
    "cmc": 5
  }
}
```

**Response:**
```json
{
  "cardData": "Flying. When this creature enters...",
  "imageData": "base64-encoded-image"
}
```

## 🐛 Troubleshooting

### CORS Errors
- Make sure Flask server is running with CORS enabled
- Check that Angular is connecting to `http://localhost:5000`

### Ollama Connection Failed
```bash
# Make sure Ollama is running
ollama serve

# Check if model is installed
ollama list

# Install model if needed
ollama pull mistral:latest
```

### Port Conflicts
- **Angular**: Default port 4200
- **Flask**: Default port 5000  
- **Ollama**: Default port 11434

Change ports if needed:
```bash
ng serve --port 4201
# or
python app.py  # Edit app.py to change Flask port
```

## 🔧 Built With

- **[Angular 16](https://angular.io/)** - Frontend framework
- **[Flask](https://flask.palletsprojects.com/)** - Backend API
- **[Ollama](https://ollama.com/)** - Local AI text generation
- **[Angular Material](https://material.angular.io/)** - UI components
- **[Mana Font](https://mana.andrewgioia.com/)** - MTG mana symbols

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Magic: The Gathering is a trademark of Wizards of the Coast
- Mana symbols provided by [Mana Font](https://mana.andrewgioia.com/)
- AI text generation powered by [Ollama](https://ollama.com/)

---

*Create amazing Magic: The Gathering cards powered by AI! ✨*