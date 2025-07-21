# TradeSight Backend

A comprehensive stock trading and prediction platform backend built with FastAPI, featuring machine learning-powered stock price predictions, portfolio management, and real-time market data integration.

## 🚀 Features

- **Stock Price Prediction**: Advanced ML models using LSTM, GRU, and Transformer architectures
- **Portfolio Management**: Complete trading simulation with buy/sell operations
- **Real-time Market Data**: Integration with Yahoo Finance for live stock data
- **User Authentication**: JWT-based secure authentication system
- **Watchlist Management**: Track favorite stocks and market movements
- **Technical Analysis**: Custom technical indicators (ADX, RSI, MACD, etc.)
- **Transaction History**: Complete audit trail of all trading activities
- **Market Analytics**: Top movers, indices, and market cap data

## 📁 Project Structure

```
tradesight-backend/
├── app/                          # Main application directory
│   ├── api/                      # API layer
│   │   ├── endpoints/            # API endpoint definitions
│   │   │   ├── account.py        # Account management endpoints
│   │   │   ├── market.py         # Market data endpoints
│   │   │   ├── prediction.py     # ML prediction endpoints
│   │   │   ├── transaction.py    # Trading transaction endpoints
│   │   │   ├── users.py          # User authentication endpoints
│   │   │   └── wishlist.py       # Watchlist management endpoints
│   │   └── __init__.py           # API router configuration
│   ├── core/                     # Core application components
│   │   ├── config.py             # Application configuration
│   │   └── security.py           # Authentication & security utilities
│   ├── database/                 # Database layer
│   │   ├── database.py           # Database connection & session management
│   │   └── models.py             # SQLAlchemy ORM models
│   ├── models_store/             # Pre-trained ML models storage
│   │   └── *.h5                  # TensorFlow/Keras model files for NIFTY 50 stocks
│   ├── schemas/                  # Pydantic data models
│   │   ├── account.py            # Account-related schemas
│   │   ├── prediction.py         # Prediction request/response schemas
│   │   ├── profit.py             # Profit calculation schemas
│   │   ├── stock_models.py       # Stock data schemas
│   │   ├── symbol.py             # Stock symbol schemas
│   │   ├── token.py              # Authentication token schemas
│   │   ├── transaction.py        # Transaction schemas
│   │   └── user.py               # User management schemas
│   ├── services/                 # Business logic layer
│   │   ├── helper_service.py     # Utility functions
│   │   ├── mover_service.py      # Market data service
│   │   ├── prediction_service.py # ML prediction service
│   │   ├── stock_service.py      # Stock data service
│   │   └── transaction_service.py # Transaction processing service
│   └── main.py                   # FastAPI application entry point
├── mydatabase.db                 # SQLite database file
├── pyproject.toml                # Project dependencies & configuration
├── uv.lock                       # Dependency lock file
├── .gitignore                    # Git ignore rules
├── .python-version               # Python version specification
└── README.md                     # Project documentation
```

## 🏗️ Architecture Overview & Detailed Folder Structure

### 📁 Root Directory Files

- **`mydatabase.db`**: SQLite database file containing all application data
- **`pyproject.toml`**: Project configuration with dependencies and metadata
- **`uv.lock`**: Dependency lock file ensuring reproducible builds
- **`.gitignore`**: Git ignore rules for excluding unnecessary files
- **`.python-version`**: Specifies Python 3.12 requirement
- **`README.md`**: This documentation file

### 📁 `app/` - Main Application Directory

#### 📁 `app/core/` - Core Configuration & Security

- **`config.py`**:
  - Environment variable management
  - Database URL configuration
  - JWT settings (SECRET_KEY, ALGORITHM, TOKEN_EXPIRY)
  - Debug mode configuration
- **`security.py`**:
  - JWT token creation and validation
  - Password hashing utilities
  - User authentication functions
  - Access token management

#### 📁 `app/database/` - Database Layer

- **`database.py`**:
  - SQLAlchemy engine configuration
  - Database session management
  - Connection pooling setup
  - Database dependency injection
- **`models.py`**:
  - **User Model**: Authentication, cash balance, relationships
  - **Portfolio Model**: User stock holdings with quantities and average prices
  - **Watchlist Model**: User's tracked stocks
  - **Transaction Model**: Complete trading history with timestamps
  - **StockIndex Model**: Market indices (NIFTY 50, etc.)
  - **IndexStock Model**: Stocks belonging to specific indices

#### 📁 `app/api/` - API Layer

- **`__init__.py`**:
  - API router configuration
  - Endpoint registration
  - URL prefix setup (/api/v1)

##### 📁 `app/api/endpoints/` - API Endpoints

- **`users.py`**:
  - User registration and login
  - JWT token generation
  - User profile management
  - Authentication endpoints
- **`market.py`**:
  - Top movers (gainers/losers)
  - Market indices data
  - Candlestick/OHLC data
  - Market announcements and news
  - Market cap information
  - Order book data
- **`account.py`**:
  - Portfolio management
  - Account balance operations
  - Holdings summary
  - Performance analytics
- **`transaction.py`**:
  - Buy/sell order execution
  - Transaction history
  - Trade confirmation
  - Order management
- **`wishlist.py`**:
  - Add/remove stocks from watchlist
  - Watchlist retrieval
  - Stock tracking management
- **`prediction.py`**:
  - ML model predictions
  - Technical analysis
  - Price forecasting
  - Model performance metrics

#### 📁 `app/services/` - Business Logic Layer

- **`prediction_service.py`**:
  - **CustomTechnicalIndicators Class**: ADX, RSI, MACD, Bollinger Bands
  - **ML Model Loading**: TensorFlow/Keras model management
  - **Data Preprocessing**: Feature engineering and scaling
  - **Prediction Pipeline**: End-to-end prediction workflow
  - **Technical Analysis**: Advanced indicator calculations
- **`mover_service.py`**:
  - Yahoo Finance API integration
  - Top gainers/losers identification
  - Market indices tracking
  - Real-time price data
  - Market cap calculations
- **`stock_service.py`**:
  - Stock data retrieval and processing
  - Historical data management
  - Price validation
  - Symbol lookup and verification
- **`transaction_service.py`**:
  - Trade execution logic
  - Portfolio updates
  - Balance management
  - Transaction validation
  - Order processing
- **`helper_service.py`**:
  - Utility functions
  - Common calculations
  - Data formatting helpers
  - Validation utilities

#### 📁 `app/schemas/` - Pydantic Data Models

- **`user.py`**:
  - User registration/login schemas
  - User profile models
  - Authentication request/response models
- **`account.py`**:
  - Portfolio schemas
  - Account balance models
  - Holdings representation
- **`transaction.py`**:
  - Buy/sell order schemas
  - Transaction history models
  - Trade confirmation schemas
- **`stock_models.py`**:
  - Stock data representation
  - OHLC data models
  - Market data schemas
- **`prediction.py`**:
  - Prediction request/response models
  - ML model output schemas
  - Technical analysis results
- **`symbol.py`**:
  - Stock symbol validation
  - Ticker representation
  - Market identifier schemas
- **`token.py`**:
  - JWT token schemas
  - Authentication response models
- **`profit.py`**:
  - Profit/loss calculation models
  - Performance metrics schemas

#### 📁 `app/models_store/` - Pre-trained ML Models

Contains 50 pre-trained TensorFlow/Keras models (\*.h5 files) for NIFTY 50 stocks:

- **Individual Stock Models**: Each NIFTY 50 stock has a dedicated model
- **Model Architecture**: LSTM, GRU, and Transformer-based networks
- **Features**: Technical indicators, price patterns, volume analysis
- **Examples**:
  - `RELIANCE.NS_model.h5` - Reliance Industries model
  - `TCS.NS_model.h5` - Tata Consultancy Services model
  - `HDFCBANK.NS_model.h5` - HDFC Bank model
  - And 47 more NIFTY 50 stock models

#### 📁 `app/main.py` - Application Entry Point

- **FastAPI Application**: Main app instance creation
- **CORS Middleware**: Cross-origin request handling
- **Router Integration**: API endpoint registration
- **Database Migration**: Automatic schema updates
- **Application Startup**: Database table creation

### 🔄 Data Flow Architecture

1. **Request Flow**: Client → API Endpoints → Services → Database
2. **Authentication Flow**: Login → JWT Token → Protected Routes
3. **Prediction Flow**: Stock Symbol → ML Service → Model → Prediction
4. **Trading Flow**: Order → Validation → Execution → Portfolio Update
5. **Market Data Flow**: Yahoo Finance → Service → Cache → Client

### 🗄️ Database Relationships

```
Users (1) ←→ (N) Portfolio
Users (1) ←→ (N) Watchlist
Users (1) ←→ (N) Transactions
StockIndex (1) ←→ (N) IndexStock
```

### 🔧 Service Dependencies

- **Prediction Service**: TensorFlow, Scikit-learn, Technical Indicators
- **Market Service**: Yahoo Finance API, Real-time data
- **Transaction Service**: Database operations, Portfolio management
- **Stock Service**: Data validation, Symbol verification

## 🛠️ Technology Stack

- **Framework**: FastAPI (Python 3.12+)
- **Database**: SQLite with SQLAlchemy ORM
- **ML/AI**: TensorFlow, Keras, Scikit-learn
- **Data Processing**: Pandas, NumPy
- **Market Data**: Yahoo Finance API
- **Authentication**: JWT tokens with python-jose
- **Visualization**: Matplotlib, Seaborn
- **Package Management**: UV (ultra-fast Python package installer)

## 📊 Database Schema

### Core Tables

- **`users`**: User accounts with authentication and cash balance
- **`portfolio`**: User stock holdings with quantities and average prices
- **`watchlist`**: User's tracked stocks
- **`transactions`**: Complete trading history
- **`stock_indices`**: Market indices (NIFTY 50, etc.)
- **`index_stocks`**: Stocks belonging to specific indices

## 🔮 Machine Learning Features

### Prediction Models

- **LSTM Networks**: For sequential pattern recognition
- **GRU Networks**: Efficient alternative to LSTM
- **Transformer Architecture**: Attention-based predictions
- **Technical Indicators**: ADX, RSI, MACD, Bollinger Bands

### Custom Technical Indicators

- **ADX (Average Directional Index)**: Trend strength measurement
- **RSI (Relative Strength Index)**: Momentum oscillator
- **MACD**: Moving average convergence divergence
- **Bollinger Bands**: Volatility indicators
- **Moving Averages**: Various timeframes

## 🚀 Getting Started

### Prerequisites

- Python 3.12+
- UV package manager

### Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd tradesight-backend
   ```

2. **Install dependencies**

   ```bash
   uv sync
   ```

3. **Set up environment variables**
   Create a `.env` file in the root directory:

   ```env
   DATABASE_URL=sqlite:///./mydatabase.db
   SECRET_KEY=your-secret-key-here
   DEBUG=False
   ALGORITHM=HS256
   ACCESS_TOKEN_EXPIRE_MINUTES=30
   ```

4. **Run the application**

   ```bash
   uv run uvicorn app.main:app --reload
   ```

5. **Access the API**
   - API Documentation: `http://localhost:8000/docs`
   - Alternative Docs: `http://localhost:8000/redoc`

## 📡 API Endpoints

### Authentication

- `POST /api/v1/login` - User authentication
  - **Request**: `{"username": "string", "password": "string"}`
  - **Response**: `{"token": "jwt_token", "message": "Login Successful!", "clientID": "uuid"}`
- `POST /api/v1/register` - User registration
  - **Request**: `{"username": "string", "email": "string", "password": "string"}`
  - **Response**: User creation confirmation

### Market Data

- `POST /api/v1/topmovers` - Get top gaining/losing stocks
  - **Request**: `{"size": 10}`
  - **Response**: List of top performing stocks
- `POST /api/v1/getIndices` - Fetch market indices
  - **Response**: NIFTY 50 and other major indices data
- `POST /api/v1/getCandles` - Historical price data
  - **Request**: `{"symbol": "RELIANCE.NS"}`
  - **Response**: OHLC candlestick data
- `POST /api/v1/announcements` - Market news and announcements
  - **Response**: Latest market news and corporate announcements
- `POST /api/v1/getMarketCap` - Market capitalization data
  - **Response**: Company valuation information
- `POST /api/v1/getOrderBook` - Order book data
  - **Request**: `{"symbol": "RELIANCE.NS"}`
  - **Response**: Bid/ask order book information

### Trading

- `POST /api/v1/buyScrip` - Execute buy orders
  - **Request**: `{"scrip": "RELIANCE.NS", "quantity": 10, "price": 2500.0}`
  - **Response**: Trade confirmation and updated portfolio
- `POST /api/v1/sellScrip` - Execute sell orders
  - **Request**: `{"scrip": "RELIANCE.NS", "quantity": 5, "price": 2550.0}`
  - **Response**: Trade confirmation and updated portfolio
- `GET /api/v1/portfolio` - Get user portfolio
  - **Response**: Complete portfolio holdings with P&L
- `GET /api/v1/transactions` - Transaction history
  - **Response**: Complete trading history with timestamps

### Predictions

- `POST /api/v1/predict` - Get ML-based price predictions
  - **Request**: `{"symbol": "RELIANCE.NS", "days": 30}`
  - **Response**: Price predictions with technical indicators
- `GET /api/v1/models` - Available prediction models
  - **Response**: List of available NIFTY 50 stock models

### Watchlist

- `POST /api/v1/addWatchlist` - Add stock to watchlist
  - **Request**: `{"scrip": "RELIANCE.NS"}`
  - **Response**: Updated watchlist
- `POST /api/v1/removeWatchList` - Remove from watchlist
  - **Request**: `{"scrip": "RELIANCE.NS"}`
  - **Response**: Updated watchlist
- `GET /api/v1/watchlist` - Get user watchlist
  - **Response**: Complete watchlist with current prices

## 🔧 Configuration

### Environment Variables

- `DATABASE_URL`: Database connection string (default: sqlite:///./mydatabase.db)
- `SECRET_KEY`: JWT signing key for token generation
- `DEBUG`: Enable debug mode (default: False)
- `ALGORITHM`: JWT algorithm (default: HS256)
- `ACCESS_TOKEN_EXPIRE_MINUTES`: Token expiration time (default: 30)

### Model Configuration

- Models are stored in `app/models_store/`
- Each NIFTY 50 stock has a dedicated trained model
- Models use standardized input features and technical indicators
- Model files are approximately 1.4MB each (TensorFlow/Keras .h5 format)

### External API Configuration

- **Foursight Backend API**: `https://foursight-backend.harshiyer.workers.dev/api/v1`
- **Yahoo Finance**: Used for additional market data through yfinance library
- **Timeout Settings**: 10 seconds for external API calls

## 🔧 Detailed Implementation

### 🔐 Authentication System

- **JWT-based Authentication**: Secure token-based authentication with configurable expiration
- **User Registration**: Username, email, and password validation with constraints
- **Login System**: Direct password comparison with token generation
- **Client ID Generation**: Unique UUID for each session
- **Protected Routes**: Middleware-based route protection

### 💰 Trading System

- **Virtual Trading**: Simulated trading environment with $250,000 starting balance
- **Buy/Sell Operations**: Complete order execution with portfolio updates
- **Portfolio Management**: Real-time holdings tracking with average prices
- **Transaction History**: Complete audit trail with timestamps
- **Balance Management**: Automatic cash balance updates

### 📊 Market Data Integration

- **External API Integration**: Foursight backend API for real-time data
- **Top Movers**: Real-time gainers and losers identification
- **Market Indices**: NIFTY 50 and other major indices tracking
- **Candlestick Data**: OHLC data for technical analysis
- **Market Cap Data**: Company valuation information
- **Order Book**: Bid/ask data for trading decisions

### 🤖 Machine Learning Pipeline

- **50 Pre-trained Models**: Individual models for each NIFTY 50 stock
- **Technical Indicators**: Custom implementation of ADX, RSI, MACD, Bollinger Bands
- **Data Preprocessing**: StandardScaler and PCA for feature engineering
- **Model Architecture**: LSTM, GRU, and Transformer networks
- **Prediction API**: Real-time price prediction endpoints

### 📈 Technical Analysis Features

- **ADX (Average Directional Index)**: Trend strength measurement
- **RSI (Relative Strength Index)**: Momentum oscillator (0-100 scale)
- **MACD**: Moving Average Convergence Divergence
- **Bollinger Bands**: Volatility-based trading bands
- **Moving Averages**: Multiple timeframe support

### 🗄️ Database Design

```sql
-- Users table with authentication and balance
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    username STRING UNIQUE,
    email STRING UNIQUE,
    hashed_password STRING,
    cash_balance FLOAT DEFAULT 250000.0
);

-- Portfolio holdings
CREATE TABLE portfolio (
    id INTEGER PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    scrip STRING,
    quantity INTEGER,
    buy_price FLOAT
);

-- Transaction history
CREATE TABLE transactions (
    id INTEGER PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    scrip STRING,
    transaction_type STRING, -- 'BUY' or 'SELL'
    quantity INTEGER,
    price FLOAT,
    timestamp DATETIME
);

-- Watchlist tracking
CREATE TABLE watchlist (
    id INTEGER PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    scrip STRING
);
```

## 🧪 Development

### Code Structure

- **Schemas**: Pydantic models for request/response validation
- **Services**: Business logic separated from API endpoints
- **Models**: Database ORM models with relationships
- **Security**: JWT-based authentication with role management

### Adding New Features

1. Define schemas in `app/schemas/`
2. Create service logic in `app/services/`
3. Add API endpoints in `app/api/endpoints/`
4. Update database models if needed

## 📈 Performance Features

- **Async/Await**: Non-blocking I/O operations
- **Connection Pooling**: Efficient database connections
- **Caching**: Model and data caching for faster responses
- **Batch Processing**: Efficient bulk operations

## 🔒 Security Features

- **JWT Authentication**: Secure token-based auth
- **Password Hashing**: Secure password storage
- **CORS Configuration**: Cross-origin request handling
- **Input Validation**: Pydantic schema validation
- **SQL Injection Protection**: ORM-based queries

## 🚀 Deployment

### Production Setup

1. Set `DEBUG=False` in environment
2. Use production database (PostgreSQL recommended)
3. Configure proper SECRET_KEY
4. Set up reverse proxy (Nginx)
5. Use process manager (PM2, systemd)

### Docker Deployment

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN pip install uv && uv sync
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:

- Create an issue in the repository
- Check the API documentation at `/docs`
- Review the code examples in the endpoints

## 🔮 Future Enhancements

- Real-time WebSocket connections
- Advanced portfolio analytics
- Options trading support
- Social trading features
- Mobile app integration
- Advanced charting capabilities
