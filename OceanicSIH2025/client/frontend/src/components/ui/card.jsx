// import * as React from "react"

// export function Card({ className = "", children }) {
//   return (
//     <div className={`bg-white rounded-2xl shadow-md ${className}`}>
//       {children}
//     </div>
//   )
// }

// export function CardContent({ className = "", children }) {
//   return (
//     <div className={`p-4 ${className}`}>
//       {children}
//     </div>
//   )
// }
import * as React from "react"

export function Card({ className = "", children }) {
  return (
    <div className={`bg-white rounded-2xl shadow-md border border-gray-200 ${className}`}>
      {children}
    </div>
  )
}

export function CardHeader({ className = "", children }) {
  return (
    <div className={`p-4 border-b border-gray-200 ${className}`}>
      {children}
    </div>
  )
}

export function CardTitle({ className = "", children }) {
  return (
    <h2 className={`text-lg font-semibold ${className}`}>
      {children}
    </h2>
  )
}

export function CardContent({ className = "", children }) {
  return (
    <div className={`p-4 ${className}`}>
      {children}
    </div>
  )
}

export function CardFooter({ className = "", children }) {
  return (
    <div className={`p-4 border-t border-gray-200 ${className}`}>
      {children}
    </div>
  )
}
