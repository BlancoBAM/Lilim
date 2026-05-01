import { ChatInterface } from './components/ChatInterface';

export default function App() {
  return (
    <div className="min-h-screen bg-gray-950 flex items-center justify-center p-8">
      <div className="w-full max-w-lg h-[90vh]">
        <ChatInterface />
      </div>
    </div>
  );
}