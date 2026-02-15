import { AppProvider } from './context'
import RouteMap from './components/RouteMap'
import Sidebar from './components/Sidebar'
import ElevationProfile from './components/ElevationProfile'
import ZoneForm from './components/ZoneForm'

function App() {
  return (
    <AppProvider>
      <div className="h-screen w-screen relative overflow-hidden">
        <RouteMap />
        <Sidebar />
        <ElevationProfile />
        <ZoneForm />
      </div>
    </AppProvider>
  )
}

export default App
