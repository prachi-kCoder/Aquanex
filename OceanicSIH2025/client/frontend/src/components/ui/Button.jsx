import * as React from "react"

export function Button({ className = "", ...props }) {
  return (
    <button
      className={`px-4 py-2 bg-blue-600 text-white font-medium rounded-xl hover:bg-blue-700 transition ${className}`}
      {...props}
    />
  )
}
