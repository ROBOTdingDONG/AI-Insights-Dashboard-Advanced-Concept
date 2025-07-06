{
  "name": "ai-insights-dashboard-frontend",
  "version": "1.0.0",
  "description": "AI Insights Dashboard - Modern React TypeScript Frontend",
  "private": true,
  "homepage": ".",
  "engines": {
    "node": ">=18.0.0",
    "npm": ">=8.0.0"
  },
  "scripts": {
    "start": "react-scripts start",
    "build": "react-scripts build",
    "test": "react-scripts test",
    "eject": "react-scripts eject",
    "lint": "eslint src --ext .ts,.tsx,.js,.jsx",
    "lint:fix": "eslint src --ext .ts,.tsx,.js,.jsx --fix",
    "format": "prettier --write \"src/**/*.{ts,tsx,js,jsx,json,css,md}\"",
    "format:check": "prettier --check \"src/**/*.{ts,tsx,js,jsx,json,css,md}\"",
    "type-check": "tsc --noEmit",
    "analyze": "npm run build && npx bundle-analyzer build/static/js/*.js",
    "preview": "serve -s build -l 3000",
    "clean": "rm -rf build node_modules",
    "docker:build": "docker build -t ai-insights-frontend .",
    "docker:run": "docker run -p 3000:3000 ai-insights-frontend"
  },
  "dependencies": {
    "@types/node": "^20.10.5",
    "@types/react": "^18.2.45",
    "@types/react-dom": "^18.2.18",
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-scripts": "5.0.1",
    "typescript": "^5.3.3",
    
    "react-router-dom": "^6.20.1",
    "@types/react-router-dom": "^5.3.3",
    
    "recharts": "^2.8.0",
    "@types/recharts": "^1.8.29",
    
    "axios": "^1.6.2",
    
    "react-hook-form": "^7.48.2",
    "@hookform/resolvers": "^3.3.2",
    "zod": "^3.22.4",
    
    "react-query": "^3.39.3",
    "@tanstack/react-query": "^5.14.2",
    "@tanstack/react-query-devtools": "^5.14.2",
    
    "date-fns": "^3.0.6",
    "lodash": "^4.17.21",
    "@types/lodash": "^4.14.202",
    
    "clsx": "^2.0.0",
    "class-variance-authority": "^0.7.0",
    
    "react-hot-toast": "^2.4.1",
    "react-loading-skeleton": "^3.3.1",
    "react-intersection-observer": "^9.5.3",
    
    "lucide-react": "^0.300.0",
    "framer-motion": "^10.16.16",
    
    "js-cookie": "^3.0.5",
    "@types/js-cookie": "^3.0.6",
    
    "react-markdown": "^9.0.1",
    "remark-gfm": "^4.0.0",
    "react-syntax-highlighter": "^15.5.0",
    "@types/react-syntax-highlighter": "^15.5.11",
    
    "react-dropzone": "^14.2.3",
    "file-saver": "^2.0.5",
    "@types/file-saver": "^2.0.7",
    
    "recharts-to-png": "^2.3.1",
    "html2canvas": "^1.4.1",
    "jspdf": "^2.5.1",
    
    "react-helmet-async": "^2.0.4",
    "react-error-boundary": "^4.0.11",
    
    "zustand": "^4.4.7",
    "immer": "^10.0.3"
  },
  "devDependencies": {
    "@testing-library/jest-dom": "^6.1.5",
    "@testing-library/react": "^14.1.2",
    "@testing-library/user-event": "^14.5.1",
    
    "tailwindcss": "^3.3.6",
    "autoprefixer": "^10.4.16",
    "postcss": "^8.4.32",
    "@tailwindcss/forms": "^0.5.7",
    "@tailwindcss/typography": "^0.5.10",
    "@tailwindcss/aspect-ratio": "^0.4.2",
    
    "eslint": "^8.56.0",
    "@typescript-eslint/eslint-plugin": "^6.15.0",
    "@typescript-eslint/parser": "^6.15.0",
    "eslint-config-prettier": "^9.1.0",
    "eslint-plugin-prettier": "^5.1.2",
    "eslint-plugin-react": "^7.33.2",
    "eslint-plugin-react-hooks": "^4.6.0",
    "eslint-plugin-jsx-a11y": "^6.8.0",
    "eslint-plugin-import": "^2.29.1",
    
    "prettier": "^3.1.1",
    "prettier-plugin-tailwindcss": "^0.5.9",
    
    "husky": "^8.0.3",
    "lint-staged": "^15.2.0",
    
    "msw": "^2.0.11",
    
    "webpack-bundle-analyzer": "^4.10.1",
    "serve": "^14.2.1",
    
    "@types/testing-library__jest-dom": "^6.0.0",
    
    "cross-env": "^7.0.3",
    "concurrently": "^8.2.2"
  },
  "browserslist": {
    "production": [
      ">0.2%",
      "not dead",
      "not op_mini all"
    ],
    "development": [
      "last 1 chrome version",
      "last 1 firefox version",
      "last 1 safari version"
    ]
  },
  "eslintConfig": {
    "extends": [
      "react-app",
      "react-app/jest",
      "@typescript-eslint/recommended",
      "prettier"
    ],
    "plugins": [
      "@typescript-eslint",
      "react-hooks",
      "jsx-a11y",
      "import"
    ],
    "rules": {
      "@typescript-eslint/no-unused-vars": ["error", { "argsIgnorePattern": "^_" }],
      "@typescript-eslint/explicit-function-return-type": "off",
      "@typescript-eslint/explicit-module-boundary-types": "off",
      "@typescript-eslint/no-explicit-any": "warn",
      "react-hooks/rules-of-hooks": "error",
      "react-hooks/exhaustive-deps": "warn",
      "jsx-a11y/anchor-is-valid": "off",
      "import/order": [
        "error",
        {
          "groups": [
            "builtin",
            "external",
            "internal",
            "parent",
            "sibling",
            "index"
          ],
          "newlines-between": "always",
          "alphabetize": {
            "order": "asc",
            "caseInsensitive": true
          }
        }
      ]
    },
    "settings": {
      "import/resolver": {
        "typescript": {}
      }
    }
  },
  "prettier": {
    "semi": true,
    "singleQuote": true,
    "tabWidth": 2,
    "trailingComma": "es5",
    "printWidth": 100,
    "plugins": ["prettier-plugin-tailwindcss"]
  },
  "husky": {
    "hooks": {
      "pre-commit": "lint-staged"
    }
  },
  "lint-staged": {
    "src/**/*.{ts,tsx,js,jsx}": [
      "eslint --fix",
      "prettier --write"
    ],
    "src/**/*.{json,css,md}": [
      "prettier --write"
    ]
  },
  "proxy": "http://localhost:8000",
  "jest": {
    "collectCoverageFrom": [
      "src/**/*.{ts,tsx}",
      "!src/**/*.d.ts",
      "!src/index.tsx",
      "!src/reportWebVitals.ts"
    ]
  },
  "msw": {
    "workerDirectory": "public"
  },
  "volta": {
    "node": "18.19.0",
    "npm": "10.2.3"
  },
  "packageManager": "npm@10.2.3",
  "author": {
    "name": "AI Insights Team",
    "email": "team@aiinsights.dev"
  },
  "license": "MIT",
  "repository": {
    "type": "git",
    "url": "https://github.com/ai-insights/dashboard-frontend.git"
  },
  "bugs": {
    "url": "https://github.com/ai-insights/dashboard-frontend/issues"
  },
  "keywords": [
    "react",
    "typescript",
    "dashboard",
    "ai",
    "analytics",
    "visualization",
    "search",
    "insights"
  ]
}
