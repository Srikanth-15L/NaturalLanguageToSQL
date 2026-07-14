# DataInsight AI - Frontend Dashboard

A modern, responsive web dashboard built with React, Vite, and Tailwind-inspired styling. It connects to the FastAPI backend to provide an interactive interface for chatting with the database agent and visualizing its reasoning steps.

## Features

- **Interactive Schema Browser**: Displays the available database tables, column names, types, and primary key indicators directly in the sidebar.
- **Collapsible Reasoning Tree**: Inspect the agent's real-time reasoning loops, including intermediate thoughts, chosen tools, arguments, and raw results for every query.
- **Sleek UI/UX**: Dark mode styling with clean layouts, loading indicators, and smooth transitions.

---

## Setup & Running

Make sure the FastAPI backend (`api.py`) is running on `http://127.0.0.1:8000` before starting the frontend.

### 1. Install Dependencies

Install the packages using npm:

```bash
npm install
```

### 2. Start the Development Server

Launch the Vite dev server:

```bash
npm run dev
```

This will spin up the app, typically at `http://localhost:5173/`.

### 3. Build for Production

To generate the optimized production build:

```bash
npm run build
```

The output files will be built into the `dist/` directory.

---

## Technical Details

- **Framework**: React 18+ & Vite
- **HTTP Client**: Axios (configured to call the local FastAPI backend)
- **Icons**: Lucide React
- **Styling**: Vanilla CSS (`index.css` / `App.css`) supporting smooth transitions and variables.
