import * as React from "react"

export function Select({ value, onValueChange, children }) {
  return (
    <select
      value={value}
      onChange={(e) => onValueChange(e.target.value)}
      className="w-full px-3 py-2 border rounded-xl shadow-sm focus:outline-none focus:ring focus:border-blue-400"
    >
      {children}
    </select>
  )
}

export function SelectItem({ value, children }) {
  return <option value={value}>{children}</option>
}

export function SelectTrigger({ children }) {
  return <>{children}</>
}

export function SelectValue({ placeholder }) {
  return <option value="">{placeholder}</option>
}

export function SelectContent({ children }) {
  return <>{children}</>
}
