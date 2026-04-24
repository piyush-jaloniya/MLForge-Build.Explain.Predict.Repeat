import { type ReactNode } from 'react'
import { ChevronDown } from 'lucide-react'
import { cn } from '../lib/utils'

export interface DropdownOption {
    value: string
    label: string
    disabled?: boolean
}

interface DropdownSelectProps {
    value: string
    onChange: (value: string) => void
    options: DropdownOption[]
    placeholder?: string
    ariaLabel: string
    icon?: ReactNode
    disabled?: boolean
    className?: string
    buttonClassName?: string
    menuClassName?: string
    optionClassName?: string
}

export default function DropdownSelect({
    value,
    onChange,
    options,
    placeholder = 'Select an option',
    ariaLabel,
    icon,
    disabled = false,
    className,
    buttonClassName,
    menuClassName,
    optionClassName,
}: DropdownSelectProps) {
    const hasMatchingValue = options.some(option => option.value === value)
    const showPlaceholder = !hasMatchingValue && placeholder.trim().length > 0
    const selectValue = hasMatchingValue ? value : ''

    // Intentionally keep these props in the component API for compatibility with existing callers.
    void menuClassName

    return (
        <div className={cn('relative', className)}>
            {icon && (
                <span className="pointer-events-none absolute left-3 top-1/2 z-10 -translate-y-1/2 text-foreground/65">
                    {icon}
                </span>
            )}
            <select
                aria-label={ariaLabel}
                disabled={disabled}
                value={selectValue}
                onChange={event => onChange(event.target.value)}
                className={cn(
                    'w-full appearance-none rounded-xl border border-border bg-surface py-2.5 pr-10 text-sm font-medium text-foreground shadow-sm transition-all',
                    'hover:border-brand-500/50 focus:outline-none focus:border-brand-500 disabled:cursor-not-allowed disabled:opacity-60',
                    icon ? 'pl-9' : 'pl-3',
                    buttonClassName
                )}
            >
                {showPlaceholder && (
                    <option value="" disabled>
                        {placeholder}
                    </option>
                )}
                {options.map(option => (
                    <option
                        key={option.value}
                        value={option.value}
                        disabled={option.disabled}
                        className={optionClassName}
                    >
                        {option.label}
                    </option>
                ))}
            </select>
            <ChevronDown size={16} className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 text-foreground/55" />
        </div>
    )
}