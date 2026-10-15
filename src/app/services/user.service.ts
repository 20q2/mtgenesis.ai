import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { BehaviorSubject, Observable, Subject, tap } from 'rxjs';
import { User } from '../models/api.model';
import { api } from './api.util';

export const USER_STORAGE_KEY = 'mtgenesis.user';
/** Per user: they've seen the profile picture prompt on this device (docs/.../2026-10-14-profile-pictures-design.md). */
export const AVATAR_PROMPT_KEY_PREFIX = 'mtgenesis.avatarPrompt.';

/**
 * Username-only identity (spec §7). The logged-in user lives in localStorage so a
 * refresh keeps you signed in; every storage access is guarded because private
 * browsing and blocked storage can make localStorage throw.
 */
@Injectable({ providedIn: 'root' })
export class UserService {
  /** In-memory copy, used when localStorage is unavailable. */
  private memoryUser: User | null = null;
  private readonly userSubject: BehaviorSubject<User | null>;
  private readonly loginSubject = new Subject<User>();
  /** Prompts already seen, when localStorage is unavailable. */
  private readonly promptedInMemory = new Set<string>();

  readonly user$: Observable<User | null>;
  /** Each successful login (the app offers a profile picture right after one). */
  readonly loggedIn$: Observable<User> = this.loginSubject.asObservable();

  constructor(private http: HttpClient) {
    this.memoryUser = this.readStorage();
    this.userSubject = new BehaviorSubject<User | null>(this.memoryUser);
    this.user$ = this.userSubject.asObservable();
  }

  login(username: string): Observable<User> {
    return this.http.post<User>(api('/users/login'), { username: username.trim() }).pipe(
      tap(user => {
        this.store(user);
        this.loginSubject.next(user);
      })
    );
  }

  /** Re-reads me from the server (picks up a picture set on another device). */
  refresh(): Observable<User> {
    return this.http.get<User>(api('/me')).pipe(tap(user => this.storeIfCurrent(user)));
  }

  /** Uploads a framed picture (a small JPEG); returns me with the new avatarUrl. */
  setAvatar(picture: Blob): Observable<User> {
    const headers = new HttpHeaders({ 'Content-Type': picture.type || 'image/jpeg' });
    return this.http.put<User>(api('/me/avatar'), picture, { headers })
      .pipe(tap(user => this.storeIfCurrent(user)));
  }

  removeAvatar(): Observable<User> {
    return this.http.delete<User>(api('/me/avatar')).pipe(tap(user => this.storeIfCurrent(user)));
  }

  /** True once this user has seen the profile picture prompt on this device. */
  wasPrompted(userId: string): boolean {
    try {
      return localStorage.getItem(AVATAR_PROMPT_KEY_PREFIX + userId) !== null
        || this.promptedInMemory.has(userId);
    } catch {
      return this.promptedInMemory.has(userId);
    }
  }

  markPrompted(userId: string): void {
    this.promptedInMemory.add(userId);
    try {
      localStorage.setItem(AVATAR_PROMPT_KEY_PREFIX + userId, '1');
    } catch {
      // storage unavailable: remembered for this session only
    }
  }

  currentUser(): User | null {
    const stored = this.readStorage();
    if (stored) {
      return stored;
    }
    return this.storageWorks() ? null : this.memoryUser;
  }

  logout(): void {
    this.memoryUser = null;
    try {
      localStorage.removeItem(USER_STORAGE_KEY);
    } catch {
      // storage unavailable: the in-memory copy is already cleared
    }
    this.userSubject.next(null);
  }

  /** A server copy of me, unless I logged out (or in as someone else) meanwhile. */
  private storeIfCurrent(user: User): void {
    if (this.currentUser()?.id === user.id) {
      this.store(user);
    }
  }

  private store(user: User): void {
    const clean: User = { id: user.id, username: user.username, avatarUrl: user.avatarUrl ?? null };
    this.memoryUser = clean;
    try {
      localStorage.setItem(USER_STORAGE_KEY, JSON.stringify(clean));
    } catch {
      // storage unavailable: keep the in-memory copy for this session
    }
    this.userSubject.next(clean);
  }

  private readStorage(): User | null {
    try {
      const raw = localStorage.getItem(USER_STORAGE_KEY);
      if (!raw) {
        return null;
      }
      const parsed = JSON.parse(raw);
      if (parsed && typeof parsed.id === 'string' && typeof parsed.username === 'string') {
        const avatarUrl = typeof parsed.avatarUrl === 'string' ? parsed.avatarUrl : null;
        return { id: parsed.id, username: parsed.username, avatarUrl };
      }
      return null;
    } catch {
      return null;
    }
  }

  private storageWorks(): boolean {
    try {
      localStorage.getItem(USER_STORAGE_KEY);
      return true;
    } catch {
      return false;
    }
  }
}
